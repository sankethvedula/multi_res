require "torch"
require "nn"
require "cunn"
require "image"
-- Enter the file number
--file_number = 1083

for file_number = 1,1000 do
print(file_number)
input_1_filename = "./data/data/"..(file_number)..".png"
input_2_filename = "./data/data/"..(file_number+2)..".png"
output_filename = "./data/data/"..(file_number+1)..".png"

input_1_image = image.load(input_1_filename,1,'byte')
input_2_image = image.load(input_2_filename,1,'byte')
output_image = image.load(output_filename,1,'byte')


  input_1 = image.scale(input_1_image,350,350):double()
  input_2 = image.scale(input_2_image,350,350):double()
  --output = image.scale(output_image,348,348):double():mul(2./255.):add(-1):cuda()
  disp_out = input_1 + input_2
  disp_out = disp_out/2



  disp_out = disp_out:byte()
  image.save("./ave_out/"..(file_number)..".png",disp_out)

end
