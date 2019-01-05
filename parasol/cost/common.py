from abc import abstractmethod, ABCMeta

class CostFunction(object, metaclass=ABCMeta):

    def __init__(self, ds, da):
        self.ds, self.da = ds, da
    

    @abstractmethod
    def cost_fn(self, states, actions):
        pass

    @abstractmethod
    def loss_fn(self, states, actions):
        pass

    @abstractmethod
    def get_parameters(self):
        pass

    @property
    @abstractmethod
    def is_learned(self):
        pass

    def __getstate__(self):
        return {
            'ds': self.ds,
            'da': self.da,
        }

    def __setstate__(self, state):
        self.__init__(state['ds'], state['da'])
