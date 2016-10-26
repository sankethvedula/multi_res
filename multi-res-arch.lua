require "nn"
require "optim"
require "torch"
require "cunn"


function rgb2gray(im)
	-- Image.rgb2y uses a different weight mixture

	local dim, w, h = im:size()[1], im:size()[2], im:size()[3]
	if dim ~= 3 then
		 print('<error> expected 3 channels')
		 return im
	end

	-- a cool application of tensor:select
	local r = im:select(1, 1)
	local g = im:select(1, 2)
	local b = im:select(1, 3)

	local z = torch.Tensor(w, h):zero()

	-- z = z + 0.21r
	z = z:add(0.21, r)
	z = z:add(0.72, g)
	z = z:add(0.07, b)
	return z
end


local function branch_conv_net(input_channels, output_channels,stride,pooling)
    branch = nn.Sequential()
    branch:add(nn.SpatialConvolution(input_channels,output_channels,stride,stride,1,1,7,7))
    branch:add(nn.ReLU())
    branch:add(nn.SpatialMaxPooling(2,2))
    branch:add(nn.SpatialConvolution(output_channels,input_channels,stride,stride,1,1,7,7))
    branch:add(nn.ReLU())
    branch:add(nn.SpatialUpSamplingNearest(2))
    return branch
end

require "image"

img = image.lena()
img = rgb2gray(img)
print(img:size())
img_lena = image.scale(img,256,256)

-- Convert Lena into our size

----- input lena
lena_net = nn.Sequential()
lena_net:add(nn.View(1,1,256,256))

img_lena_input = lena_net:forward(img_lena)
print(img_lena:size())

----- Output Lena
rescale_lena = image.scale(img_lena,243,243)
print(rescale_lena:size())
lena_net = nn.Sequential()
lena_net:add(nn.View(1,1,243,243))

img_lena_output = lena_net:forward(rescale_lena)

-----------Inputs--------------
frame1 = img_lena_input:cuda()
frame2 = torch.ones(1,1,256,256):cuda()

----------Targets--------------
output = img_lena_output:cuda()

-- Give the inputs as a table
input_table = {frame1, frame2}


------------------
--network starts--

--  Branch - Frame1
branch1 = nn.Sequential()

conv1 = nn.Sequential()
conv1:add(nn.SpatialConvolution(1,16,16,16,1,1,7,7))
conv1:add(nn.ReLU())
conv1:add(nn.SpatialMaxPooling(2,2))
-- Concating upper
upper = nn.ConcatTable()
-- Children upper
child1_subconv1 = branch_conv_net(16,32,16,true)
child2_subconv1 = branch_conv_net(16,32,16,true)
shortcut_1 = nn.Sequential():add(nn.Identity())
--merge1 = nn.ParallelTable(2,1)
--merge1:add(nn.SpatialConvolution(32,64,4,4)):add(nn.ReLU())
--merge1:add(nn.SpatialConvolution(32,64,4,4)):add(nn.ReLU())

merge1 = nn.JoinTable(2)
final_conv1 = nn.Sequential()
final_conv1:add(nn.SpatialConvolution(32,1,8,8,1,1,3,3)):add(nn.ReLU()):add(nn.SpatialUpSamplingNearest(2))

upper:add(child1_subconv1)
upper:add(child2_subconv1)
upper:add(shortcut_1)

--


branch1:add(conv1)
branch1:add(upper)
branch1:add(merge1)
branch1:add(final_conv1)

--- Branch - Frame2

branch2 = nn.Sequential()

conv2 = nn.Sequential()
conv2:add(nn.SpatialConvolution(1,16,16,16,1,1,7,7))
conv2:add(nn.ReLU()):add(nn.SpatialMaxPooling(2,2))

--conv2:add(nn.SpatialMaxPooling(2,2))

lower = nn.ConcatTable()

child1_subconv2 = branch_conv_net(16,32,16,true)
child2_subconv2 = branch_conv_net(16,32,16,true)

shortcut_2 = nn.Sequential():add(nn.Identity())

merge2 = nn.JoinTable(2)
final_conv2 = nn.Sequential()
final_conv2:add(nn.SpatialConvolution(32,1,8,8,1,1,3,3)):add(nn.ReLU()):add(nn.SpatialUpSamplingNearest(2))

lower:add(child1_subconv2)
lower:add(child2_subconv2)
lower:add(shortcut_2)

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
fcn:add(nn.SpatialConvolution(2,1,4,4))
--fcn:add(nn.Linear( 921600,8*8))


fc1 = nn:Sequential()
fc1:add(par_net)
fc1:add(merge_final)
fc1:add(fcn)

fc1:cuda()
--print(fc1)
--print(fc1)
--fc1:add(nn.JoinTable(1):unp ack())
--fc1:add(nn.View(-1))
--fc1:add(nn.Linear(16*6*6,4*4))

-- Criterion

criterion = nn.MSECriterion()
criterion:cuda()

-- Get the parameters and gradients
x, dl_dx = fc1:getParameters()
x:cuda()
dl_dx:cuda()

out = fc1:forward(input_table)
--print(out)
--print(params:size())


local function feval(x_new)
  if x~=x_new then
    x:xopy(x_new)
  end

  dl_dx:zero()
  predicted_output = fc1:forward(input_table)
  --print(predicted_output:size())
  loss = criterion:forward(predicted_output, output)
  grad_outs = criterion:backward(predicted_output, output)

  grad_ins = fc1:backward(input_table,grad_outs)
  return loss, dl_dx
end

require "optim"

optim_params = {learningRate = 0.01}

print(output:size())
for i = 1,2000 do
  print(i)
  fc1:training()
  local _,errs = optim.adagrad(feval, x, optim_params)
  print(errs)

end
