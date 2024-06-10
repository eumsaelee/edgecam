from queue import Queue


class InvaildSize(Exception): pass


class PushoutQueue(Queue):

    def __init__(self, maxsize: int=0):
        super().__init__(maxsize)

    def put_nowait(self, item, push_out: bool=False):
        if push_out:
            with self.mutex:
                if self._qsize >= self.maxsize:
                    self._get()
                    self.unfinished_tasks -= 1
                self._put(item)
                self.unfinished_tasks += 1
                self.not_empty.notify()
        else:
            super().put_nowait(item)


class ResizablePushoutQueue(Queue):

    def __init__(self, maxsize: int=1):
        self._maxsize = None
        super().__init__(maxsize)

    @property
    def maxsize(self) -> int:
        return self._maxsize

    @maxsize.setter
    def maxsize(self, maxsize: int):
        new_maxsize = self._inspect(maxsize)
        old_maxsize = self._maxsize
        if not old_maxsize:
            self._maxsize = new_maxsize
        else:
            with self.mutex:
                if new_maxsize < old_maxsize:
                    for _ in range(old_maxsize - new_maxsize):
                        self._get()
                        self.unfinished_tasks -= 1
                self._maxsize = new_maxsize

    def put_nowait(self, item, push_out=False):
        if push_out:
            with self.mutex:
                if self._qsize() >= self._maxsize:
                    self._get()
                    self.unfinished_tasks -= 1
                self._put(item)
                self.unfinished_tasks += 1
                self.not_empty.notify()
        else:
            super().put_nowait(item)

    @staticmethod
    def _inspect(maxsize: int) -> int:
        if isinstance(maxsize, int) and maxsize > 0:
            return maxsize
        raise InvaildSize(
            f'Parameter maxsize must be a positive integer.')