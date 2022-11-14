import torch, random
from torch import nn

# from nowcasting.config import cfg
from models.utils import make_layers
from core.fcst_type import forecasterType


class Forecaster(nn.Module):
    def __init__(self, subnets, rnns, target_len):
        super().__init__()
        assert len(subnets) == len(rnns)

        self.blocks = len(subnets)
        self._target_len = target_len
        self._is_ashesh = True

        for index, (params, rnn) in enumerate(zip(subnets, rnns)):
            setattr(self, 'rnn' + str(self.blocks - index), rnn)
            setattr(self, 'stage' + str(self.blocks - index), make_layers(params))
        print(f'[{self.__class__.__name__}] TargetLen:{self._target_len}')

    def forward_by_stage(self, input, state, subnet, rnn):
        # the kwarg is_encoder refers to the original Ashesh-type simple GRU
        input, state_stage = rnn(input, state, seq_len=self._target_len, is_ashesh = self._is_ashesh)
        seq_number, batch_size, input_channel, height, width = input.size()
        input = torch.reshape(input, (-1, input_channel, height, width))
        input = subnet(input)
        input = torch.reshape(input, (seq_number, batch_size, input.size(1), input.size(2), input.size(3)))
        
        return input

    def forward(self, hidden_states):
        input = self.forward_by_stage(None, hidden_states[-1], getattr(self, 'stage3'), getattr(self, 'rnn3'))
        for i in list(range(1, self.blocks))[::-1]: #= for i in range (2,0,-1) =(2, 1)
            input = self.forward_by_stage(input, hidden_states[i - 1], getattr(self, 'stage' + str(i)),
                                          getattr(self, 'rnn' + str(i)))

        assert input.shape[2] == 1, 'Finally, there should be only one channel'
        return input[:, :, 0, :, :]
 
class Forecaster_PONI(nn.Module):
    def __init__(self, subnets, rnns, target_len, teach):
        super().__init__()
        assert len(subnets) == len(rnns)

        self.blocks = len(subnets)
        self._target_len = target_len
        self._is_PONI = True
        self.teacher_forcing_ratio = teach

        for index, (params, rnn) in enumerate(zip(subnets, rnns)):
            setattr(self, 'rnn' + str(self.blocks - index), rnn)
            setattr(self, 'stage' + str(self.blocks - index), make_layers(params))
        print(f'[{self.__class__.__name__}] TargetLen:{self._target_len} TeacherForcing:{self.teacher_forcing_ratio}')

    def forward_by_stage(self, input, state, subnet, rnn):
        state_stage = rnn(input, state, seq_len=self._target_len, is_PONI=self._is_PONI)
        output = subnet(state_stage) # [B, C, H, W]
        return output, state_stage

    def forward(self, hidden_states, y_add, y_encoder):
        # y -> conv2d
        y_add = torch.permute(y_add, (1, 0, 2, 3)).unsqueeze(2) # [3, B, 1, H, W]
        tar_len, b, ch, h, w = y_add.size()
        y_add = torch.reshape(y_add, (-1, ch, h, w))
        y_input = y_encoder(y_add)
        y_input = torch.reshape(y_input, 
                                (tar_len, b, y_input.size(1), y_input.size(2), y_input.size(3)),
                               )

        # recurrent network
        hidden_states.reverse()
        y_container = []
        if self.teacher_factor(self.teacher_forcing_ratio): # teahcer
            for i in range(tar_len):
                for j in range(3, 0, -1): # GRU_layers
                    if j == 3:
                        output = y_input[i]
                    output, s_state = self.forward_by_stage(output,
                                                            hidden_states[3 - j], 
                                                            getattr(self, 'stage'+str(j)), 
                                                            getattr(self, 'rnn'+str(j))
                                                        )
                    hidden_states.append(s_state)
                [hidden_states.pop(0) for _ in range(3)] # according to GRU layers
                y_container.append(output)
        else: # student
            for i in range(tar_len):
                for j in range(3, 0, -1): # GRU_layers
                    if j == 3 and i == 0:
                        output = y_input[i]
                    elif j == 3:
                        output = y_encoder(output)
                    output, s_state = self.forward_by_stage(output,
                                                            hidden_states[3 - j], 
                                                            getattr(self, 'stage'+str(j)), 
                                                            getattr(self, 'rnn'+str(j))
                                                        )
                    hidden_states.append(s_state)
                [hidden_states.pop(0) for _ in range(3)] # according to GRU layers
                y_container.append(output)
        
        y_output = torch.stack(y_container) #[3, B, C, H, W]
        assert y_output.shape[2] == 1, 'Finally, there should be only one channel'
        return y_output[:, :, 0, :, :]

    def teacher_factor(self, ratio):
        return True if random.random() < ratio else False
    
# #use hetero data
# class Forecaster(nn.Module):
#     def __init__(self, subnets, rnns, target_len, teach):
#         super().__init__()
#         assert len(subnets) == len(rnns)

#         self.blocks = len(subnets)
#         self._target_len = target_len
#         self._is_decoder = True
#         self.teacher_forcing_ratio = teach
#         # self.unpool = nn.MaxUnpool2d(2, stride=2)
#         # self.unpool_indices = torch.tensor([[[[5, 7], [13, 15]]]]).cuda() # [1, 1, 2, 2]

#         for index, (params, rnn) in enumerate(zip(subnets, rnns)):
#             setattr(self, 'rnn' + str(self.blocks - index), rnn)
#             setattr(self, 'stage' + str(self.blocks - index), make_layers(params))
#         print(f'[{self.__class__.__name__}] TargetLen:{self._target_len} TeacherForcing:{self.teacher_forcing_ratio}')

#     def forward_by_stage(self, input, state, subnet, rnn):
#         state_stage = rnn(input, state, seq_len=self._target_len, is_decoder=self._is_decoder)
#         output = subnet(state_stage) # [B, C, H, W]
#         return output, state_stage

#     # def forward(self, hidden_states, y_add, y_encoder, era_encoder, era5, dateTime):
#     def forward(self, hidden_states, y_add, y_encoder, era5, dateTime):
#         ''' y -> conv2d '''
#         y_add = torch.permute(y_add, (1, 0, 2, 3)).unsqueeze(2) # [3, B, 1, H, W]
#         tar_len, b, ch, h, w = y_add.size()
#         y_add = torch.reshape(y_add, (-1, ch, h, w))
#         y_input = y_encoder(y_add)
#         y_input = torch.reshape(y_input, 
#                                 (tar_len, b, y_input.size(1), y_input.size(2), y_input.size(3)),
#                                )
#         ''' era5 data input '''
#         # era5 = torch.mean(era5[:,:,:2], dim=[4, 5]) # [B, 3, 2, 20]
#         # era5 = torch.reshape(era5, (-1, 40))
#         # era5 = era_encoder(era5)
#         # era5 = torch.reshape(era5, (tar_len, b, 4, 4)).unsqueeze(2) # customized [3, b, 1, H', W']

#         ''' datetime & season [B, 3, 2, 2]'''
#         # indices = self.concat(self.unpool_indices, 3, 1) # [1, 3, 2, 2]
#         # indices = self.concat(indices, b, 0) # [b, 3, 2, 2]
#         # dateTime = torch.permute(self.unpool(dateTime, indices), 
#                                 #  (1, 0, 2, 3)
#                                 #  ).unsqueeze(2) # [3, b, 1, H', W']
#         # dateTime = torch.permute(dateTime, (1, 0, 2, 3)).view(tar_len, b, -1)
#         # dateTime = self.uniform_map(dateTime, 4, 4) # [3, b, 4, H', W']

#         ''' combine data '''
#         # y_input = torch.cat((y_input, era5, dateTime), dim=2)
#         # y_input = torch.cat((y_input, dateTime), dim=2)

#         # recurrent network
#         hidden_states.reverse()
#         y_container = []
#         if self.teacher_factor(self.teacher_forcing_ratio): # teahcer
#             for i in range(tar_len):
#                 for j in range(3, 0, -1): # GRU_layers
#                     if j == 3:
#                         output = y_input[i]
#                     output, s_state = self.forward_by_stage(output,
#                                                             hidden_states[3 - j], 
#                                                             getattr(self, 'stage'+str(j)), 
#                                                             getattr(self, 'rnn'+str(j))
#                                                         )
#                     hidden_states.append(s_state)
#                 [hidden_states.pop(0) for _ in range(3)] # according to GRU layers
#                 y_container.append(output)
#         else: # student
#             for i in range(tar_len):
#                 for j in range(3, 0, -1): # GRU_layers
#                     if j == 3 and i == 0:
#                         output = y_input[i]
#                     elif j == 3:
#                         output = y_encoder(output)
#                         # output = torch.cat((output, era5[i], dateTime[i]), dim=1)
#                         # output = torch.cat((output, dateTime[i]), dim=1)
#                     output, s_state = self.forward_by_stage(output,
#                                                             hidden_states[3 - j], 
#                                                             getattr(self, 'stage'+str(j)), 
#                                                             getattr(self, 'rnn'+str(j))
#                                                         )
#                     hidden_states.append(s_state)
#                 [hidden_states.pop(0) for _ in range(3)] # according to GRU layers
#                 y_container.append(output)
        
#         y_output = torch.stack(y_container) #[3, B, C, H, W]
#         assert y_output.shape[2] == 1, 'Finally, there should be only one channel'
#         return y_output[:, :, 0, :, :]

#     def teacher_factor(self, ratio):
#         return True if random.random() < ratio else False

#     def concat(self, inp, concat_times:int, concat_dim:int):
#         out = [inp for _ in range(concat_times)]
#         out = torch.cat(out, dim=concat_dim)
#         assert out.shape[concat_dim] == concat_times
#         return out

#     def uniform_map(self, inp, goal_H, goal_W):
#         '''
#         inp size = [tar_len, batch_size, 4] # 4 means four datetime parameters
#         goal_H and goal_W are corresponding to the y_input
#         '''
#         new_map = torch.ones((inp.size(0), inp.size(1), inp.size(2), goal_H, goal_W)).cuda()
#         for t in range(inp.size(0)):
#             for b in range(inp.size(1)):
#                 for i in range(inp.size(2)):
#                     new_map[t,b,i] = new_map[t,b,i] * inp[t,b,i]
#         return new_map