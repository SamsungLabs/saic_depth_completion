class Registry(dict):
    def register(self, name):
        if name in self.keys():
            raise ValueError("Registry already contains such key: {}".format(name))

        def _register(fn):
            self.update({name: fn})
            return fn

        return _register


MODELS = Registry()
BACKBONES = Registry()

