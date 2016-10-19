require "nn"
require "cunn"
require "torch"
require "multi-res-model"
require "optim"
require "image"

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

fc1 = multi_res_model()
--print(fc1)
print("Loaded the network")
-- Setting the parameters
x, dl_dx = fc1:getParameters()
print(x:size())
print("Got the parameters")
x:cuda()
dl_dx:cuda()

train_data = torch.load("train_data.t7")
print("Loaded Training Data")


local function get_data(i)
  --input_1_filename = "./data/data/"..(i)..".png"
  --input_2_filename = "./data/data/"..(i+2)..".png"
  --output_filename = "./data/data/"..(i+1)..".png"
	input_1_filename = i
	input_2_filename = i+2
	output_filename = i+1
  --print(i)
  --if i == 1 then
    --input_1_image = image.load(input_1_filename,1,'byte')
		input_1_image = train_data[input_1_filename]
		--input_2_image = image.load(input_2_filename,1,'byte')
		input_2_image = train_data[input_2_filename]
		output_image = train_data[output_filename]

		--output_image = image.load(output_filename,1,'byte')
  --else
    --input_1_image:copy(output_image)
    --output_image:copy(input_2_image)
    --image_2 = image.load(input_2_filename,1,'byte')
  --end

  input_1 = image.scale(input_1_image,350,350):double():mul(2./255.):add(-1):cuda()
  input_2 = image.scale(input_2_image,350,350):double():mul(2./255.):add(-1):cuda()
  --input_1 = image.scale(input_1,350,350):double():mul(1./255.):cuda()
  --input_2 = image.scale(input_2,350,350):double():mul(1./255.):cuda()

  --image.display(input_1)
  --image.display(input_2)

  output = image.scale(output_image,348,348):double():mul(2./255.):add(-1):cuda()
  --output = image.scale(output,348,348):double():mul(1./255.):cuda()
  --image.display(output)

  --input_1 = torch.ones(350,350):cuda()
  --input_2 = torch.zeros(350,350):cuda()

  --output = torch.ones(348,348):cuda()

  return input_1,input_2,output
end



local function processed_data(input_1,input_2,output)
  --input_1 = torch.ones(350,350):cuda()
  --input_2 = torch.zeros(350,350):cuda()

  --output = torch.ones(348,348):cuda()


  -- Net that resizes our data
  resize_net_input = nn.Sequential()
  resize_net_input:add(nn.View(1,1,350,350))
  resize_net_input:cuda()

  resize_net_output = nn.Sequential()
  resize_net_output:add(nn.View(1,1,348,348))
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
        --torch.ones(1,1,350,350):cuda(),
        ---torch.ones(1,1,350,350):cuda()
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

  for number = 1,7000 do
    input_1, input_2, output = get_data(number)
    --print("got the data")
    input_table, output = processed_data(input_1,input_2,output)
    --print("processed_data")

    --print(output:size())
    local _,errs = optim.sgd(feval, x, optim_params)
    --print(errs[1])
    total_err = total_err + errs[1]
  end
  --print(total_err/100)

  return total_err/7000
end


for i = 1,10 do
  --print(i)
  fc1:training()
  total_err = single_epoch(x,dl_dx)
  print("Epoch number  "..i.."  Training Error:  "..total_err)

end

torch.save("Multi_ResNet.t7",fc1)


output_sample = fc1:forward{torch.ones(1,1,100,100):cuda(),torch.zeros(1,1,100,100):cuda()}
--print(output)
