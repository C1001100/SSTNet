import torch


class SparseL1(torch.nn.Module):
    def __init__(self, lamd=1e-1):
        super(SparseL1, self).__init__()


        self.fidelity = torch.nn.L1Loss()



    def forward(self, y, gt,loss_sparsity):
        loss1 = self.fidelity(y, gt)
        # print('lossL1',loss,loss.size())

        # print(encodeResult_up.size())
        print('loss_sparsity', loss_sparsity)


        a = 0.00001


        loss_sparsity=a * loss_sparsity
        print('loss_sparsity(', loss_sparsity, '   loss(', loss1)

       

        loss = loss1 + loss_sparsity
        # print('loss', loss)


        return loss,loss_sparsity,loss1

    def sparsityNrom(self, encodeResult):

        # print(encodeResult)
        loss_sparsity = torch.linalg.matrix_norm(encodeResult, ord=1)
        # print(loss_sparsity, loss_sparsity.size())
        loss_sparsity = torch.linalg.vector_norm(loss_sparsity, ord=1)
        # print(loss_sparsity, loss_sparsity.size())
        # loss_sparsity = torch.linalg.norm(loss_sparsity, ord=1)



        # loss_sparsity=torch.norm(encodeResult, 1)

        return loss_sparsity


