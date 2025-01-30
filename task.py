from typing import Any, Optional
from collections.abc import Callable


class Context(object):
    class _Task(object):
        def __init__(self, ctx: 'Context', name: str, job: Callable[..., Any], deps: list[str]):
            self.name = name
            self.job = job
            self.deps = deps
            self.ctx = ctx
            self.result = None

        def __repr__(self):
            return f'Task(name={self.name}, deps=[{[dep for dep in self.deps]}])'

        def __call__(self):
            return self.ctx.get(self.name)

    def __init__(self):
        self._tasks: dict[str, Context._Task] = {}

    def add_task(self, name: str, job: Callable[..., Any], deps: Optional[list[str | _Task]] = None):
        if name in self._tasks:
            raise ValueError(f"Task {name} already registered.")
        if not deps:
            deps = []
        deps = [dep.name if isinstance(dep, Context._Task) else dep for dep in deps]
        for dep in deps:
            if dep not in self._tasks:
                raise ValueError(f"Dependency {dep} does not exist.")
        self._tasks[name] = Context._Task(self, name, job, deps)
        return self._tasks[name]

    def get(self, name: str):
        task = self._tasks.get(name)
        if task is None:
            raise ValueError(f"Task {name} does not exist.")
        if task.result is None:
            deps = []
            for dep in self._tasks[name].deps:
                deps.append(self.get(dep))
            task.result = task.job(*deps)
        return task.result
