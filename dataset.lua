require "nn"
require "cunn"
require "torch"
require "multi-res-model"
require "optim"
require "image"

-- To do list
-- 1. Add the loggers properly,
-- 2. Change the dimensions, log the graphs
-- 3. Build an evaluation module

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

-- Loggers

train_logger = optim.Logger('train_error.log')
validation_logger = optim.Logger('validation_error.log')
super_train_logger = optim.Logger('super_train_logger.log')

train_logger:setNames{'train_error'}
validation_logger:setNames{'validation_error'}

super_train_logger:setNames{'super_train_error'}

-- Import Network
fc1 = multi_res_model()
--print(fc1)
print("Loaded the network")
-- Setting the parameters
x, dl_dx = fc1:getParameters()
print(x:size())
print("Got the parameters")
x:cuda()
dl_dx:cuda()

-- Loading the data (both train and test)
-- This is global, so don't have to pass it to the network, I guess, lemme check
train_data = torch.load("train_data.t7")
validation_data = torch.load("validation_data.t7")
print("Loaded Training and Validation Data")


local function get_data(i,data)
  --input_1_filename = "./data/data/"..(i)..".png"
  --input_2_filename = "./data/data/"..(i+2)..".png"
  --output_filename = "./data/data/"..(i+1)..".png"
	input_1_filename = i
	input_2_filename = i+2
	output_filename = i+1
  --print(i)
  --if i == 1 then
    --input_1_image = image.load(input_1_filename,1,'byte')
		input_1_image = data[input_1_filename]
		--input_2_image = image.load(input_2_filename,1,'byte')
		input_2_image = data[input_2_filename]
		output_image = data[output_filename]

		--output_image = image.load(output_filename,1,'byte')
  --else
    --input_1_image:copy(output_image)
    --output_image:copy(input_2_image)
    --image_2 = image.load(input_2_filename,1,'byte')
  --end

  input_1 = image.scale(input_1_image,400,400):double():mul(2./255.):add(-1):cuda()
  input_2 = image.scale(input_2_image,400,400):double():mul(2./255.):add(-1):cuda()
  --input_1 = image.scale(input_1,400,400):double():mul(1./255.):cuda()
  --input_2 = image.scale(input_2,400,400):double():mul(1./255.):cuda()

  --image.display(input_1)
  --image.display(input_2)

  output = image.scale(output_image,400,400):double():mul(2./255.):add(-1):cuda()
  --output = image.scale(output,400,400):double():mul(1./255.):cuda()
  --image.display(output)

  --input_1 = torch.ones(400,400):cuda()
  --input_2 = torch.zeros(400,400):cuda()

  --output = torch.ones(400,400):cuda()

  return input_1,input_2,output
end



local function processed_data(input_1,input_2,output)
  --input_1 = torch.ones(400,400):cuda()
  --input_2 = torch.zeros(400,400):cuda()

  --output = torch.ones(400,400):cuda()


  -- Net that resizes our data
  resize_net_input = nn.Sequential()
  resize_net_input:add(nn.View(1,1,400,400))
  resize_net_input:cuda()

  resize_net_output = nn.Sequential()
  resize_net_output:add(nn.View(1,1,400,400))
  resize_net_output:cuda()

  input_1 = resize_net_input:forward(input_1)
  input_2 = resize_net_input:forward(input_2)

  output = resize_net_output:forward(output)

  input_data = {input_1,input_2}

  return input_data, output
end

--input_table, output = get_data()
--pred_out = fc1:forward(input_table)


criterion = nn.MSECriterion()
criterion:cuda()


local function single_epoch(x,dl_dx)
  --input_table, output = processed_data()
  --print(input_table)
    --print("Inside Single Epoch")
    local function feval(x_new)
      if x~=x_new then
        x:copy(x_new)
      end
      --input_table = {
        --torch.ones(1,1,400,400):cuda(),
        ---torch.ones(1,1,400,400):cuda()
      --}
      --print(input_table)
      dl_dx:zero()
      predicted_output = fc1:forward(input_table)
      --print(predicted_output:size())
      loss = criterion:forward(predicted_output, output)
      grad_outs = criterion:backward(predicted_output, output)

      grad_ins = fc1:backward(input_table,grad_outs)
      return loss, dl_dx
    end


  optim_params = {learningRate = 0.01}
  total_err = 0

  no_of_examples = 1000000

  for number = 1,9998 do
    input_1, input_2, output = get_data(number,train_data)
    --print("got the data")
    input_table, output = processed_data(input_1,input_2,output)
    --print("processed_data")

    --print(output:size())
    local _,errs = optim.adam(feval, x, optim_params)
    --print(errs[1])
		super_train_logger:add{errs[1]}
		total_err = total_err + errs[1]
  end
  --print(total_err/100)

  return total_err/9998
end

local function validation_epoch(validation_data)
	validation_loss = 0
	for num = 1,1000 do
		input_1, input_2, output = get_data(num,validation_data)
    --print("got the data")
    input_table, output = processed_data(input_1,input_2,output)
		predicted_output = fc1:forward(input_table)
		--print(predicted_output:size())
		loss = criterion:forward(predicted_output, output)
		validation_loss = validation_loss + loss

	end
	validation_loss = validation_loss/100
	return validation_loss
end



for i = 1,100 do
  --print(i)
  fc1:training()
  total_err = single_epoch(x,dl_dx)
	training_loss = tostring(total_err)
  print("Epoch number  "..i.."  Training Error:  "..total_err)
	if i%20 == 0 then
		torch.save("Multi_ResNet_adagrad_itr_"..i..".t7",fc1)
	end
	fc1:evaluate()
	validation_err = validation_epoch(validation_data)
	validation_err = tostring(validation_err)
	train_logger:add{training_loss}
	validation_logger:add{validation_err}

end

train_logger:plot()
validation_logger:plot()
super_train_logger:plot()

torch.save("Multi_ResNet_adagrad.t7",fc1)


output_sample = fc1:forward{torch.ones(1,1,100,100):cuda(),torch.zeros(1,1,100,100):cuda()}
--print(output)
