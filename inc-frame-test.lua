require "torch"
require "cunn"
require "nn"
require "optim"
require "image"

i = 80
net_name = "Multi_ResNet_adagrad_itr_"..i..".t7"
net = torch.load(net_name)

count = 0

for file_number = 1,1000 do
  print(file_number)
  input_1_filename = "./data/data_low_frame/"..(file_number)..".png"
  --input_2_filename = "./data/data/"..(file_number+2)..".png"
  input_2_filename = "./data/data_low_frame/"..(file_number+1)..".png"

  input_1_image = image.load(input_1_filename,1,'byte')
  input_2_image = image.load(input_2_filename,1,'byte')

  input_1 = image.scale(input_1_image,400,400):double():mul(2./255.):add(-1):cuda()
  input_2 = image.scale(input_2_image,400,400):double():mul(2./255.):add(-1):cuda()
  --output = image.scale(output_image,400,400):double():mul(2./255.):add(-1):cuda()


    resize_net_input = nn.Sequential()
    resize_net_input:add(nn.View(1,1,400,400))
    resize_net_input:cuda()

    resize_net_output = nn.Sequential()
    resize_net_output:add(nn.View(1,1,400,400))
    resize_net_output:cuda()

    input_1 = resize_net_input:forward(input_1)
    input_2 = resize_net_input:forward(input_2)

    input_data = {input_1,input_2}

    output_net = nn.Sequential()
    output_net:add(nn.View(400,400)):cuda()

    pred_out = net:forward(input_data)
    disp_out = output_net:forward(pred_out):add(1):mul(255./2.):byte()
    input_1 = output_net:forward(input_1):add(1):mul(255./2.):byte()
    input_2 = output_net:forward(input_2):add(1):mul(255./2.):byte()

    input_2:add(1):mul(255./2.):byte()



    image.save("./low_frame_test/"..(count)..".png",input_1)
    image.save("./low_frame_test/"..(count+1)..".png",disp_out)
    image.save("./low_frame_test/"..(count+2)..".png",input_2)

    count = count+2

end
