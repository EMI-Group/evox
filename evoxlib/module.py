def lift(func):
    def wrapper(self, state, *args, **kargs):
        if self.name == '_top_level':
            return state | func(self, state, *args, **kargs)
        else:
            inner_state = state[self.name]
            new_inner_state = func(self, inner_state, *args, **kargs)
            return state | {f'{self.name}': new_inner_state}
    return wrapper

class Module:
    def setup(self):
        pass

    def init(self, name='_top_level'):
        self.name = name
        state = {}
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Module):
                submodule_name = f'_sub_{attr_name}'
                state[submodule_name] = attr.init(submodule_name)
        return state | self.setup()