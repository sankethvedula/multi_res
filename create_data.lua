require "torch"
require "image"
require "nn"
require "cunn"

folder_name = "./data/data_1/"

train_images = {}

for i = 10000,20000 do
  print(i)
  image_read = image.load(folder_name..i..".png",1,'byte')
  image_read = image.scale(image_read,350,350)
  table.insert(train_images,image_read)

end

torch.save("validation_data.t7",train_images)
--print(train_images)
