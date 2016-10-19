function multi_res_model()

require "nn"
require "optim"
require "torch"
require "cunn"




local function branch_conv_net(input_channels, output_channels,kernel_size,pooling)
    padding_size = (kernel_size-1)/2
    branch = nn.Sequential()
    --branch2 = nn.Sequential()
    branch:add(nn.SpatialConvolution(input_channels,output_channels,kernel_size,kernel_size,1,1,padding_size,padding_size))
    --branch = require('weight-init')(branch,'kaiming')
    --branch:add(branch1)
    branch:add(nn.ReLU())

    branch:add(nn.SpatialMaxPooling(2,2))

    branch:add(nn.SpatialConvolution(output_channels,input_channels,kernel_size,kernel_size,1,1,padding_size,padding_size))
    --branch2 = require('weight-init')(branch2,'kaiming')
    --branch:add(branch2)
    branch:add(nn.ReLU())
    branch:add(nn.SpatialUpSamplingNearest(2))
    branch = require('weight-init')(branch,'kaiming')
    return branch
end


input_table = {
  torch.rand(1,1,350,350):cuda(),
  torch.rand(1,1,350,350):cuda()
}
------------------
--network starts--

--  Branch - Frame1
branch1 = nn.Sequential()

conv1 = nn.Sequential()
conv1:add(nn.SpatialConvolution(1,16,7,7,1,1,3,3))
conv1:add(nn.ReLU())
conv1:add(nn.SpatialMaxPooling(2,2))
conv1 = require('weight-init')(conv1,'kaiming')
-- Concating upper
upper = nn.ConcatTable()
-- Children upper
child1_subconv1 = branch_conv_net(16,32,5,true)
child2_subconv1 = branch_conv_net(16,32,5,true)

--merge1 = nn.ParallelTable(2,1)
--merge1:add(nn.SpatialConvolution(32,64,4,4)):add(nn.ReLU())
--merge1:add(nn.SpatialConvolution(32,64,4,4)):add(nn.ReLU())

merge1 = nn.JoinTable(2)
final_conv1 = nn.Sequential()
final_conv1:add(nn.SpatialConvolution(32,1,3,3,1,1,1,1)):add(nn.ReLU()):add(nn.SpatialUpSamplingNearest(2))
final_conv1 = require('weight-init')(final_conv1,'kaiming')

upper:add(child1_subconv1)
upper:add(child2_subconv1)

--


branch1:add(conv1)
branch1:add(upper)
branch1:add(merge1)
branch1:add(final_conv1)

--- Branch - Frame2

branch2 = nn.Sequential()

conv2 = nn.Sequential()
conv2:add(nn.SpatialConvolution(1,16,7,7,1,1,3,3))
conv2:add(nn.ReLU()):add(nn.SpatialMaxPooling(2,2))
conv2 = require('weight-init')(conv2,'kaiming')

--conv2:add(nn.SpatialMaxPooling(2,2))

lower = nn.ConcatTable()

child1_subconv2 = branch_conv_net(16,32,5,true)
child2_subconv2 = branch_conv_net(16,32,5,true)

merge2 = nn.JoinTable(2)
final_conv2 = nn.Sequential()
final_conv2:add(nn.SpatialConvolution(32,1,3,3,1,1,1,1)):add(nn.ReLU()):add(nn.SpatialUpSamplingNearest(2))
final_conv2 = require('weight-init')(final_conv2,'kaiming')


lower:add(child1_subconv2)
lower:add(child2_subconv2)

branch2:add(conv2)
branch2:add(lower)
branch2:add(merge2)
branch2:add(final_conv2)

--Join the convolutional layers
--last_conv = nn.Sequential()
--last_conv:add(nn.SpatialConvolution(128,256,4,4)):add(nn.ReLU())

merge_final = nn.JoinTable(2)
par_net = nn.ParallelTable(2,1)
par_net:add(branch1)
par_net:add(branch2)
--par_net:add(merge_final)
--par_net:add(last_conv)
-- FCN

fcn = nn.Sequential()
fcn:add(nn.SpatialConvolution(2,1,5,5,1,1,2,2))
fcn = require('weight-init')(fcn,'kaiming')

--fcn:add(nn.Linear( 921600,8*8))


fc1 = nn:Sequential()
fc1:add(par_net)
fc1:add(merge_final)
fc1:add(fcn)


-- Initialization of weights


fc1:cuda()
--print(fc1)
--print(fc1)
--fc1:add(nn.JoinTable(1):unp ack())
--fc1:add(nn.View(-1))
--fc1:add(nn.Linear(16*6*6,4*4))

--for i = 1,10000 do

--0print(i)
out = fc1:forward(input_table)
print(out:size())

--print(params:size())
--print(out)
return fc1
end
