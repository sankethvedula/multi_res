require "torch"
require "nn"
require "cunn"
require "image"
-- Enter the file number
file_number = 1083

for file_number = 1,1000 do
print(file_number)
input_1_filename = "./data/data/"..(file_number)..".png"
input_2_filename = "./data/data/"..(file_number+2)..".png"
output_filename = "./data/data/"..(file_number+1)..".png"

input_1_image = image.load(input_1_filename,1,'byte')
input_2_image = image.load(input_2_filename,1,'byte')
output_image = image.load(output_filename,1,'byte')


  input_1 = image.scale(input_1_image,350,350):double():mul(2./255.):add(-1):cuda()
  input_2 = image.scale(input_2_image,350,350):double():mul(2./255.):add(-1):cuda()
  output = image.scale(output_image,348,348):double():mul(2./255.):add(-1):cuda()


  resize_net_input = nn.Sequential()
  resize_net_input:add(nn.View(1,1,350,350))
  resize_net_input:cuda()

  resize_net_output = nn.Sequential()
  resize_net_output:add(nn.View(1,1,348,348))
  resize_net_output:cuda()

  input_1 = resize_net_input:forward(input_1)
  input_2 = resize_net_input:forward(input_2)

  output = resize_net_output:forward(output)

  output_net = nn.Sequential()
  output_net:add(nn.View(348,348)):cuda()

  input_data = {input_1,input_2}

-- Let's get our network
--for i = 20,100,20 do
  i = 100
  net_name = "Multi_ResNet_itr_"..i..".t7"
  net = torch.load(net_name)
  pred_out = net:forward(input_data)
  disp_out = output_net:forward(pred_out):add(1):mul(255./2.):byte()
  image.save("./out/"..(file_number)..".png",disp_out)

end
