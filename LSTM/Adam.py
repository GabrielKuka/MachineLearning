import numpy as np

class Adam:

    def __init__(self, eps=1e-8, beta1=0.9, beta2=0.999):

        self.BETA1 = beta1
        self.BETA2 = beta2
        self.EPS = eps 

    def update_m(self, m_params, gradients):

        result = m_params*self.BETA1 + \
            (1-self.BETA1) * gradients

        return result

    def update_v(self, v_params, gradients):

        result = v_params*self.BETA2 + \
            (1-self.BETA2) * gradients**2

        return result
    
    def bias_correction(self, m_params, v_params, time_step):

        m = m_params / (1 - self.BETA1**time_step) 
        v = v_params / (1 - self.BETA2**time_step)

        return (m, v)