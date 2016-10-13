require 'torch'
require 'nn'
require 'optim'
mnist = require 'mnist'

fullset = mnist.traindataset()
testset = mnist.testdataset()


layer_size = 49
model = nn.Sequential()

model:add(nn.Reshape(28*28))
model:add(nn.Linear(28*28, layer_size))
model:add(nn.Tanh())
model:add(nn.Linear(layer_size, 28*28))
model:add(nn.Reshape(28, 28))

criterion = nn.MSECriterion()

sgd_params = {
   learningRate = 1e-2,
   learningRateDecay = 1e-4,
   weightDecay = 1e-3,
   momentum = 1e-4
}



trainset = {
    size = 50000,
    data = fullset.data[{{1,50000}}]:double(),
    label = fullset.label[{{1,50000}}]
}

validationset = {
    size = 10000,
    data = fullset.data[{{50001,60000}}]:double(),
    label = fullset.label[{{50001,60000}}]
}


x, dl_dx = model:getParameters()


step = function(batch_size)
    local current_loss = 0
    local shuffle = torch.randperm(trainset.size)
    --print(shuffle)
    batch_size = batch_size or 200

    for t = 1,trainset.size,batch_size do
        -- setup inputs for this mini-batch
        -- no need to setup targets, since they are the same
        local size = math.min(t + batch_size - 1, trainset.size) - t
        local inputs = torch.Tensor(size, 28, 28)
        for i = 1,size do
            inputs[i] = trainset.data[shuffle[i+t]]
        end

        local feval = function(x_new)
            -- reset data
            if x ~= x_new then x:copy(x_new) end
            dl_dx:zero()

            -- perform mini-batch gradient descent
            local loss = criterion:forward(model:forward(inputs), inputs)
            model:backward(inputs, criterion:backward(model.output, inputs))

            return loss, dl_dx
        end

        _, fs = optim.sgd(feval, x, sgd_params)
        -- fs is a table containing value of the loss function
        -- (just 1 value for the SGD optimization)
        current_loss = current_loss + fs[1]
    end

    return current_loss
end

--step(100)
eval = function(dataset, batch_size)
    local loss = 0
    batch_size = batch_size or 200

    for i = 1,dataset.size,batch_size do
        local size = math.min(i + batch_size - 1, dataset.size) - i
        local inputs = dataset.data[{{i,i+size-1}}]
        local outputs = model:forward(inputs)
        loss = loss + criterion:forward(model:forward(inputs), inputs)
    end

    return loss
end


max_iters = 30



do
    local last_loss = 0
    local increasing = 0
    local threshold = 1 -- how many deacreasing epochs we allow
    for i = 1,max_iters do
        local loss = step()
        print(string.format('Epoch: %d Current loss: %4f', i, loss))
        local validation_loss = eval(validationset)
        print(string.format('Loss on the validation set: %4f', validation_loss))
        if last_loss < validation_loss then
            if increasing > threshold then break end
            increasing = increasing + 1
        else
            increasing = 0
        end
        last_loss = validation_loss
    end
end
