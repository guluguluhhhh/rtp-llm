import gc
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel

from rtp_llm.access_logger.access_logger import AccessLogger
from rtp_llm.async_decoder_engine.base_engine import BaseEngine
from rtp_llm.config.engine_config import EngineConfig, update_worker_addrs
from rtp_llm.config.log_config import get_log_path
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.distribute.distributed_server import DistributedServer, get_world_info
from rtp_llm.metrics import kmonitor
from rtp_llm.model_factory import ModelFactory
from rtp_llm.models_py.distributed.collective_torch import (
    destroy_distributed_environment,
    init_distributed_environment,
)
from rtp_llm.utils.concurrency_controller import get_global_controller
from rtp_llm.utils.fuser import _nfs_manager

StreamObjectType = Union[Dict[str, Any], BaseModel]

USAGE_HEADER = "USAGE"


class BackendManager(object):
    def __init__(self, py_env_configs: PyEnvConfigs):
        self.py_env_configs = py_env_configs
        self._access_logger = AccessLogger(
            get_log_path(),
            py_env_configs.profiling_debug_logging_config.log_file_backup_count,
            py_env_configs.server_config.rank_id,
            py_env_configs.server_config.frontend_server_id,
        )
        self._distributed_server = DistributedServer(py_env_configs)
        self.thread_lock_ = threading.Lock()
        self._global_controller = get_global_controller()
        # just rank 0 report metric
        if py_env_configs.parallelism_config.world_rank == 0:
            kmonitor.init()
        self.engine: Optional[BaseEngine] = None
        self._shutdown_requested = threading.Event()
        self._nccl_init_args = None

    def start(self):
        """Initialize backend server without entering service loop"""
        self._distributed_server.start(self.py_env_configs)
        engine_config = EngineConfig.create(
            self.py_env_configs,
            nccl_comm_config=self._distributed_server.get_nccl_comm_config(),
        )

        # NCCL init in background thread — no data dependency with model config
        # creation or weight loading, so they run in parallel.
        nccl_thread = None
        nccl_error = [None]

        if engine_config.parallelism_config.world_size > 1:

            def _init_nccl():
                try:
                    init_distributed_environment(
                        engine_config.parallelism_config,
                        nccl_comm_config=self._distributed_server.get_nccl_comm_config(),
                        nccl_init_port=self._distributed_server.get_nccl_init_port(),
                        backend="nccl",
                        timeout=self.py_env_configs.distribute_config.dist_comm_timeout,
                    )
                except Exception as e:
                    nccl_error[0] = e

            nccl_thread = threading.Thread(target=_init_nccl, daemon=True)
            nccl_thread.start()
            self._nccl_init_args = {
                "parallelism_config": engine_config.parallelism_config,
                "nccl_comm_config": self._distributed_server.get_nccl_comm_config(),
                "default_port": self._distributed_server.get_nccl_init_port(),
                "timeout": self.py_env_configs.distribute_config.dist_comm_timeout,
            }

        # These steps don't depend on NCCL — run while NCCL initializes.
        world_info = get_world_info(
            self.py_env_configs.server_config,
            self.py_env_configs.distribute_config,
            self.py_env_configs.parallelism_config,
            distributed_server=self._distributed_server,
        )
        update_worker_addrs(
            engine_config.runtime_config,
            engine_config.parallelism_config,
            world_info,
        )
        model_config = ModelFactory.create_model_config(
            model_args=self.py_env_configs.model_args,
            lora_config=self.py_env_configs.lora_config,
            kv_cache_config=engine_config.kv_cache_config,
            profiling_debug_logging_config=engine_config.profiling_debug_logging_config,
            generate_env_config=self.py_env_configs.generate_env_config,
            embedding_config=self.py_env_configs.embedding_config,
            quantization_config=self.py_env_configs.quantization_config,
            render_config=self.py_env_configs.render_config,
            eplb_config=self.py_env_configs.eplb_config,
        )
        ModelFactory.update_engine_config_from_model_config(
            engine_config=engine_config,
            model_config=model_config,
        )

        # Weight loading doesn't need NCCL — still parallel with NCCL init.
        model = ModelFactory._create_model(
            model_config=model_config,
            engine_config=engine_config,
            vit_config=self.py_env_configs.vit_config,
            merge_lora=self.py_env_configs.lora_config.merge_lora,
        )

        # Wait for NCCL before anything that needs collective communication.
        if nccl_thread is not None:
            nccl_thread.join()
            if nccl_error[0] is not None:
                raise nccl_error[0]

        # DeepEP/MoriEP depends on NCCL process groups.
        if (
            model_config.expert_num > 0
            and engine_config.parallelism_config.world_size > 1
            and not engine_config.moe_config.use_all_gather
        ):
            deepep_init_success = False
            moriep_init_success = False

            if engine_config.moe_config.use_deepep_moe:
                try:
                    from rtp_llm.models_py.distributed.deepep_wrapper import (
                        init_deepep_wrapper,
                    )

                    init_deepep_wrapper(engine_config, model_config)
                    deepep_init_success = True
                except Exception as e:
                    logging.error(f"Failed to initialize DeepEP wrapper: {e}")

            if engine_config.moe_config.use_mori_ep:
                try:
                    from rtp_llm.models_py.distributed.moriep_wrapper import (
                        init_moriep_wrapper,
                    )

                    init_moriep_wrapper(engine_config, model_config)
                    moriep_init_success = True
                    logging.info("MoriEP wrapper initialized successfully")
                except Exception as e:
                    logging.error(f"Failed to initialize MoriEP wrapper: {e}")

            if engine_config.moe_config.use_deepep_moe and not deepep_init_success:
                raise RuntimeError("DeepEP was requested but failed to initialize")
            if engine_config.moe_config.use_mori_ep and not moriep_init_success:
                raise RuntimeError(
                    "use_mori_ep is set but MoriEP wrapper failed to initialize"
                )

        propose_model_config = ModelFactory.create_propose_model_config(
            engine_config=engine_config,
            model_config=model_config,
            model_args=self.py_env_configs.model_args,
        )

        # Engine init + warmup needs both NCCL and model weights.
        self.engine = ModelFactory.from_model_configs(
            model=model,
            model_config=model_config,
            engine_config=engine_config,
            world_info=world_info,
            vit_config=self.py_env_configs.vit_config,
            propose_model_config=propose_model_config,
        )
        logging.info(
            "engine created successfully: self.engine.task_type=%s",
            self.engine.task_type,
        )

    def serve_forever(self):
        gc.collect()
        gc.freeze()
        self._start_ckpt_watchdog()
        logging.info("BackendManager entering serve_forever loop")
        while not self._shutdown_requested.is_set():
            time.sleep(0.1)
        logging.info("Shutdown requested, stopping BackendManager...")
        self.stop()
        logging.info("BackendManager stopped successfully")

    def _start_ckpt_watchdog(self):
        rank = self.py_env_configs.parallelism_config.world_rank
        cmd_path = Path(f"/tmp/rtp_llm_ckpt_cmd_rank{rank}")
        ack_path = Path(f"/tmp/rtp_llm_ckpt_ack_rank{rank}")
        cmd_path.unlink(missing_ok=True)
        ack_path.unlink(missing_ok=True)

        def loop():
            import torch

            while not self._shutdown_requested.is_set():
                if cmd_path.exists():
                    try:
                        cmd = json.loads(cmd_path.read_text())
                    except Exception:
                        cmd_path.unlink(missing_ok=True)
                        continue
                    action = cmd.get("action", "")
                    logging.info(f"[ckpt_watchdog] rank{rank} action={action}")
                    try:
                        result = self._handle_ckpt_action(action, cmd, rank)
                        ack_path.write_text(result)
                    except Exception as e:
                        logging.exception(f"[ckpt_watchdog] action={action} failed")
                        ack_path.write_text(f"ERR:{type(e).__name__}:{e}")
                    cmd_path.unlink(missing_ok=True)
                time.sleep(0.1)

        threading.Thread(target=loop, daemon=True, name="ckpt_watchdog").start()
        logging.info(f"[ckpt_watchdog] started for rank {rank}")

    def _handle_ckpt_action(self, action, cmd, rank):
        import torch

        if action == "checkpoint_enter":
            ft_op = self.engine.rtp_llm_op_.ft_op
            ft_op.set_checkpoint_requested(True)
            # 发 dummy 推理请求 kick schedule() 解除 cond_var wait
            try:
                import urllib.request

                port = self.py_env_configs.server_config.start_port or 8088
                body = b'{"model":"x","messages":[{"role":"user","content":"_ckpt_kick"}],"max_tokens":1}'
                req = urllib.request.Request(
                    f"http://localhost:{port}/v1/chat/completions",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                urllib.request.urlopen(req, timeout=5)
            except Exception:
                pass  # kick 不需要成功，只需让 schedule() 返回
            for _ in range(1200):  # 60s timeout
                if ft_op.get_checkpoint_ready():
                    return "OK"
                time.sleep(0.05)
            return "ERR:checkpoint_ready_timeout"
        elif action == "destroy_nccl_safe":
            destroy_distributed_environment()
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(3)
            return "OK"
        elif action == "reinit_and_resume":
            if self._nccl_init_args:
                new_port = cmd.get("new_port") or (
                    self._nccl_init_args["default_port"] + 1000
                )
                init_distributed_environment(
                    self._nccl_init_args["parallelism_config"],
                    nccl_comm_config=self._nccl_init_args["nccl_comm_config"],
                    nccl_init_port=int(new_port),
                    backend="nccl",
                    timeout=self._nccl_init_args["timeout"],
                )
            ft_op = self.engine.rtp_llm_op_.ft_op
            ft_op.set_checkpoint_requested(False)
            return "OK"
        else:
            return f"ERR:unknown-action:{action}"

    def request_shutdown(self):
        """Request graceful shutdown of the backend manager"""
        logging.info("BackendManager shutdown requested")
        self._shutdown_requested.set()

    def stop(self) -> None:
        """Stop the backend manager and cleanup resources"""
        if isinstance(self.engine, BaseEngine):
            _nfs_manager.unmount_all()
            logging.info("all nfs paths unmounted")
            self.engine.stop()

    def ready(self):
        if isinstance(self.engine, BaseEngine):
            return self.engine.ready()
        return True

    @property
    def role_type(self) -> str:
        return self.engine.role_type if self.engine else "unknown"
