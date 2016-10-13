require "nn"
require "torch"
require "cunn"

-- Architecture is to Conv_Net -
--                               \ FCN - Output
--                    Conv_net -

-- inputs --
------------
--[[
Function to create a branch with a SpatialConvolution, ReLU and an optional MaxPooling
Inputs : input_channels, output_channels, stride, pooling(true/false)
Outputs : the network
]]
local function branch_conv_net(input_channels, output_channels,stride,pooling)
    branch = nn.Sequential()
    branch:add(nn.SpatialConvolution(input_channels,output_channels,stride,stride))
    branch:add(nn.ReLU())
    if pooling == true then
      branch:add(nn.SpatialMaxPooling(2,2))
    end
    return branch
end


-----------Inputs--------------
frame1 = torch.ones(100,1,16,16):cuda()
frame2 = torch.ones(100,1,16,16):cuda()

----------Targets--------------
output = torch.rand(100,1,8,8)

-- Give the inputs as a table
input_table = {frame1, frame2}


------------------
--network starts--

--  Branch - Frame1
branch1 = nn.Sequential()

conv1 = nn.Sequential()
conv1:add(nn.SpatialConvolution(1,16,4,4))
conv1:add(nn.ReLU())
--conv1:add(nn.SpatialMaxPooling(2,2))
-- Concating upper
upper = nn.ConcatTable()
-- Children upper
child1_subconv1 = branch_conv_net(16,32,4,false)
child2_subconv1 = branch_conv_net(16,32,4,false)

--merge1 = nn.ParallelTable(2,1)
--merge1:add(nn.SpatialConvolution(32,64,4,4)):add(nn.ReLU())
--merge1:add(nn.SpatialConvolution(32,64,4,4)):add(nn.ReLU())

merge1 = nn.JoinTable(2)
final_conv1 = nn.Sequential()
final_conv1:add(nn.SpatialConvolution(64,128,4,4)):add(nn.ReLU())

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
conv2:add(nn.SpatialConvolution(1,16,4,4))
conv2:add(nn.ReLU())

--conv2:add(nn.SpatialMaxPooling(2,2))

lower = nn.ConcatTable()

child1_subconv2 = branch_conv_net(16,32,4,false)
child2_subconv2 = branch_conv_net(16,32,4,false)

merge2 = nn.JoinTable(2)
final_conv2 = nn.Sequential()
final_conv2:add(nn.SpatialConvolution(64,128,4,4)):add(nn.ReLU())

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
fcn:add(nn.View(-1))
fcn:add(nn.Linear(100*256*7*7,8*8))


fc1 = nn:Sequential()
fc1:add(par_net)
fc1:add(merge_final)
fc1:add(fcn)

fc1:cuda()

--print(fc1)
--fc1:add(nn.JoinTable(1):unp ack())
--fc1:add(nn.View(-1))
--fc1:add(nn.Linear(16*6*6,4*4))

-- Criterion

criterion = nn.MSECriterion()

output = fc1:forward(input_table)


print(output:size())
