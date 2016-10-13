require 'nn'
require 'torch'
require "image"
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


img = image.lena()
img = rgb2gray(img)

img = image.scale(img,100,100)
-- Encoder
print(img:size())

layer_size = 1000
model = nn.Sequential()
model:add(nn.Reshape(img:size(1)*img:size(2)))
model:add(nn.Linear(img:size(1)*img:size(2), layer_size))
model:add(nn.Tanh())
model:add(nn.Linear(layer_size, img:size(1)*img:size(2)))
model:add(nn.Reshape(img:size(1),img:size(2)))

--[[
decoder = nn.Sequential()
decoder:add(nn.Sequential(20,40,5,5))
decoder:add(nn.ReLU())
decoder:add(nn.Sequential(40,3,5,5))
]]
net = nn.Sequential()
net:add(model)

print(net)

require "optim"
criterion = nn.MSECriterion()

optim_params = {learningRate = 0.1}
x, gradients = net:getParameters()

--img_out = image.scale(img,496,496)

-- All in CUDA()
--net:cuda()
--criterion:cuda()
--img_out:cuda()
--img:cuda()

--net:add(decoder)
function feval(x_new)
  if x~= x_new then
    x:copy(x_new)
  end

  gradients:zero()
  pred_output = net:forward(img)
  --print(pred_output:size())
  errors = criterion:forward(pred_output,img)
  grad_outputs = criterion:backward(pred_output,img)
  grad_inputs = net:backward(img,grad_outputs)
  return errors, gradients
end
-- Training Process



for i = 1, 20000 do
  print(i)
  net:training()
  local _,errs = optim.sgd(feval,x,optim_params)
  print(errs)


end
