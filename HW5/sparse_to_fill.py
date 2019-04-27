import torch
import torch.nn as nn


class PruningWeight(object):
    def __init__(self, ratio=0.0):
        self.MaskList = []
        self.threshold = []
        self.ratio = ratio

    def Init(self, model):
        for m in model.modules():            
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                with torch.no_grad():                                                    
                    m.weight.copy_(self._SetUpPruning(m.weight, self.ratio))

    def _SetUpPruning(self, weight, ratio):
        _threshold = self._FindMidValue(weight, ratio)
        sparse_weight, _Mask = self._InitMask(weight, _threshold)
        self.threshold.append(_threshold)
        self.MaskList.append(_Mask)
        return sparse_weight

    def _FindMidValue(self, weight, ratio):
        #################################
        ######## Put your codes here ####
        #################################
        
        
        
        #################################
        ######### End of your codes #####
        #################################
        return threshold

    def _InitMask(self, w, threshold):
        mask = torch.abs(w).ge(threshold).type(dtype=torch.float32)
        #################################
        ######## Put your codes here ####
        #################################
        
        
        
        #################################
        ######### End of your codes #####
        #################################
        return w, mask

    def RecoverSparse(self, model):
        _idx = 0
        for m in model.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                with torch.no_grad():
                    #################################
                    ######## Put your codes here ####
                    #################################
        
               
                    #################################
                    ######### End of your codes #####
                    #################################
                _idx += 1

    def TestSparse(self, model):
        zero_cnt = 0
        all_cnt = 0
        for m in model.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                w = m.weight
                zero_cnt += torch.sum((w == 0).int()).item()
                all_cnt += w.nelement()
        print ('the sparsity of model is %.3f' % (1.0*zero_cnt/all_cnt))
