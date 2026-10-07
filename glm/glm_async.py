"""Keep native EXL3 GPU steps off the HTTP event loop; engine sources stay intact."""
import asyncio
from concurrent.futures import ThreadPoolExecutor


def responsive_generator_class(base):
    class ResponsiveAsyncGenerator(base):
        def __init__(self, *args, **kwargs):
            self._gpu_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="glm-gpu")
            self._step_lock = asyncio.Lock()
            try:
                super().__init__(*args, **kwargs)
            except BaseException:
                self._gpu_worker.shutdown(wait=True)
                raise

        async def _run_iteration(self):
            try:
                while True:
                    async with self.condition:
                        await self.condition.wait_for(lambda: bool(self.jobs))
                    async with self._step_lock:
                        future = asyncio.get_running_loop().run_in_executor(
                            self._gpu_worker, self.generator.iterate)
                        try:
                            results = await asyncio.shield(future)
                        except asyncio.CancelledError:
                            # Do not release the lock while an uncancellable GPU
                            # step still reads the job/page table being cancelled.
                            await future
                            raise
                        self.deliver_results(results)
                    await asyncio.sleep(0)
            except asyncio.CancelledError:
                return
            except Exception as error:
                self.error = error
                for job in self.jobs.values():
                    job.put_result(error)
                self.jobs.clear()

        async def cancel(self, job):
            async with self._step_lock:
                await super().cancel(job)

        async def close(self):
            try:
                await super().close()
            finally:
                await asyncio.to_thread(self._gpu_worker.shutdown, wait=True,
                                        cancel_futures=True)

    return ResponsiveAsyncGenerator
