**Validation score for base network** - 82.89

**Model def**

model = Sequential()
model.add(SeparableConv2D(64, 3, 3, border_mode='same', input_shape=(32, 32, 3), use_bias=False)) #3 (32, 32, 64)

model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(SeparableConv2D(128, 3, 3, use_bias=False)) #5 (30, 30, 128)

model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2))) #(15, 15, 128)
model.add(Dropout(0.3))

model.add(SeparableConv2D(64, 3, 3, border_mode='same', use_bias=False)) #7 (15, 15, 64)

model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(SeparableConv2D(128, 3, 3, use_bias=False)) #11 (13, 13, 128)

model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2))) #(6, 6, 128)
model.add(Dropout(0.3))

model.add(SeparableConv2D(64, 3, 3, border_mode='same', use_bias=False)) #13 (6, 6, 64)

model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(SeparableConv2D(64, 3, 3, use_bias=False)) #15 (4, 4, 64)

model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(SeparableConv2D(32, 3, 3, use_bias=False)) #17 (2, 2, 32)

model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(SeparableConv2D(10, 2, 2, use_bias=False)) #19 (1, 1, 10)


model.add(Flatten())
model.add(Activation('softmax'))

**Log**

Epoch 1/50

Epoch 00001: LearningRateScheduler setting learning rate to 0.05.
390/390 [==============================] - 24s 62ms/step - loss: 1.5678 - acc: 0.4175 - val_loss: 4.3397 - val_acc: 0.2730
Epoch 2/50

Epoch 00002: LearningRateScheduler setting learning rate to 0.0379075057.
390/390 [==============================] - 19s 47ms/step - loss: 1.2390 - acc: 0.5545 - val_loss: 1.4729 - val_acc: 0.5475
Epoch 3/50

Epoch 00003: LearningRateScheduler setting learning rate to 0.0305250305.
390/390 [==============================] - 19s 48ms/step - loss: 1.0153 - acc: 0.6380 - val_loss: 1.3435 - val_acc: 0.5942
Epoch 4/50

Epoch 00004: LearningRateScheduler setting learning rate to 0.0255493102.
390/390 [==============================] - 19s 48ms/step - loss: 0.8786 - acc: 0.6909 - val_loss: 1.1430 - val_acc: 0.6178
Epoch 5/50

Epoch 00005: LearningRateScheduler setting learning rate to 0.0219683656.
390/390 [==============================] - 19s 48ms/step - loss: 0.7902 - acc: 0.7226 - val_loss: 1.0481 - val_acc: 0.6623
Epoch 6/50

Epoch 00006: LearningRateScheduler setting learning rate to 0.0192678227.
390/390 [==============================] - 19s 48ms/step - loss: 0.7345 - acc: 0.7411 - val_loss: 0.7748 - val_acc: 0.7371
Epoch 7/50

Epoch 00007: LearningRateScheduler setting learning rate to 0.017158545.
390/390 [==============================] - 19s 48ms/step - loss: 0.6884 - acc: 0.7577 - val_loss: 0.8413 - val_acc: 0.7227
Epoch 8/50

Epoch 00008: LearningRateScheduler setting learning rate to 0.0154655119.
390/390 [==============================] - 19s 48ms/step - loss: 0.6467 - acc: 0.7740 - val_loss: 0.8539 - val_acc: 0.7232
Epoch 9/50

Epoch 00009: LearningRateScheduler setting learning rate to 0.0140765766.
390/390 [==============================] - 19s 48ms/step - loss: 0.6238 - acc: 0.7833 - val_loss: 0.7043 - val_acc: 0.7637
Epoch 10/50

Epoch 00010: LearningRateScheduler setting learning rate to 0.012916559.
390/390 [==============================] - 19s 48ms/step - loss: 0.5940 - acc: 0.7931 - val_loss: 0.8687 - val_acc: 0.7164
Epoch 11/50

Epoch 00011: LearningRateScheduler setting learning rate to 0.0119331742.
390/390 [==============================] - 19s 48ms/step - loss: 0.5774 - acc: 0.7994 - val_loss: 0.6855 - val_acc: 0.7752
Epoch 12/50

Epoch 00012: LearningRateScheduler setting learning rate to 0.0110889332.
390/390 [==============================] - 19s 48ms/step - loss: 0.5555 - acc: 0.8059 - val_loss: 0.7367 - val_acc: 0.7562
Epoch 13/50

Epoch 00013: LearningRateScheduler setting learning rate to 0.0103562552.
390/390 [==============================] - 19s 48ms/step - loss: 0.5415 - acc: 0.8103 - val_loss: 0.6495 - val_acc: 0.7847
Epoch 14/50

Epoch 00014: LearningRateScheduler setting learning rate to 0.0097143967.
390/390 [==============================] - 19s 48ms/step - loss: 0.5200 - acc: 0.8174 - val_loss: 0.5756 - val_acc: 0.8067
Epoch 15/50

Epoch 00015: LearningRateScheduler setting learning rate to 0.009147457.
390/390 [==============================] - 19s 48ms/step - loss: 0.5071 - acc: 0.8218 - val_loss: 0.5686 - val_acc: 0.8102
Epoch 16/50

Epoch 00016: LearningRateScheduler setting learning rate to 0.0086430424.
390/390 [==============================] - 19s 48ms/step - loss: 0.4978 - acc: 0.8265 - val_loss: 0.5761 - val_acc: 0.8117
Epoch 17/50

Epoch 00017: LearningRateScheduler setting learning rate to 0.0081913499.
390/390 [==============================] - 19s 48ms/step - loss: 0.4871 - acc: 0.8302 - val_loss: 0.6086 - val_acc: 0.7971
Epoch 18/50

Epoch 00018: LearningRateScheduler setting learning rate to 0.0077845244.
390/390 [==============================] - 19s 48ms/step - loss: 0.4743 - acc: 0.8344 - val_loss: 0.5877 - val_acc: 0.8044
Epoch 19/50

Epoch 00019: LearningRateScheduler setting learning rate to 0.007416197.
390/390 [==============================] - 19s 49ms/step - loss: 0.4697 - acc: 0.8367 - val_loss: 0.5908 - val_acc: 0.8041
Epoch 20/50

Epoch 00020: LearningRateScheduler setting learning rate to 0.00708115.
390/390 [==============================] - 19s 48ms/step - loss: 0.4612 - acc: 0.8392 - val_loss: 0.6408 - val_acc: 0.7983
Epoch 21/50

Epoch 00021: LearningRateScheduler setting learning rate to 0.0067750678.
390/390 [==============================] - 19s 48ms/step - loss: 0.4518 - acc: 0.8421 - val_loss: 0.5738 - val_acc: 0.8109
Epoch 22/50

Epoch 00022: LearningRateScheduler setting learning rate to 0.0064943499.
390/390 [==============================] - 19s 48ms/step - loss: 0.4403 - acc: 0.8449 - val_loss: 0.5500 - val_acc: 0.8195
Epoch 23/50

Epoch 00023: LearningRateScheduler setting learning rate to 0.0062359691.
390/390 [==============================] - 19s 48ms/step - loss: 0.4399 - acc: 0.8451 - val_loss: 0.5706 - val_acc: 0.8134
Epoch 24/50

Epoch 00024: LearningRateScheduler setting learning rate to 0.0059973612.
390/390 [==============================] - 19s 48ms/step - loss: 0.4296 - acc: 0.8494 - val_loss: 0.5445 - val_acc: 0.8184
Epoch 25/50

Epoch 00025: LearningRateScheduler setting learning rate to 0.0057763401.
390/390 [==============================] - 19s 48ms/step - loss: 0.4239 - acc: 0.8494 - val_loss: 0.5782 - val_acc: 0.8103
Epoch 26/50

Epoch 00026: LearningRateScheduler setting learning rate to 0.0055710306.
390/390 [==============================] - 19s 48ms/step - loss: 0.4216 - acc: 0.8519 - val_loss: 0.5740 - val_acc: 0.8112
Epoch 27/50

Epoch 00027: LearningRateScheduler setting learning rate to 0.0053798149.
390/390 [==============================] - 19s 48ms/step - loss: 0.4158 - acc: 0.8536 - val_loss: 0.5740 - val_acc: 0.8166
Epoch 28/50

Epoch 00028: LearningRateScheduler setting learning rate to 0.0052012899.
390/390 [==============================] - 19s 48ms/step - loss: 0.4153 - acc: 0.8532 - val_loss: 0.5580 - val_acc: 0.8160
Epoch 29/50

Epoch 00029: LearningRateScheduler setting learning rate to 0.0050342328.
390/390 [==============================] - 19s 48ms/step - loss: 0.4051 - acc: 0.8570 - val_loss: 0.5419 - val_acc: 0.8220
Epoch 30/50

Epoch 00030: LearningRateScheduler setting learning rate to 0.0048775729.
390/390 [==============================] - 19s 48ms/step - loss: 0.4046 - acc: 0.8586 - val_loss: 0.5450 - val_acc: 0.8260
Epoch 31/50

Epoch 00031: LearningRateScheduler setting learning rate to 0.004730369.
390/390 [==============================] - 19s 48ms/step - loss: 0.3962 - acc: 0.8597 - val_loss: 0.5728 - val_acc: 0.8140
Epoch 32/50

Epoch 00032: LearningRateScheduler setting learning rate to 0.0045917899.
390/390 [==============================] - 19s 48ms/step - loss: 0.3918 - acc: 0.8615 - val_loss: 0.5397 - val_acc: 0.8259
Epoch 33/50

Epoch 00033: LearningRateScheduler setting learning rate to 0.0044610992.
390/390 [==============================] - 19s 48ms/step - loss: 0.3961 - acc: 0.8603 - val_loss: 0.5259 - val_acc: 0.8261
Epoch 34/50

Epoch 00034: LearningRateScheduler setting learning rate to 0.0043376421.
390/390 [==============================] - 19s 49ms/step - loss: 0.3918 - acc: 0.8623 - val_loss: 0.5501 - val_acc: 0.8216
Epoch 35/50

Epoch 00035: LearningRateScheduler setting learning rate to 0.004220834.
390/390 [==============================] - 19s 48ms/step - loss: 0.3874 - acc: 0.8651 - val_loss: 0.5555 - val_acc: 0.8195
Epoch 36/50

Epoch 00036: LearningRateScheduler setting learning rate to 0.0041101521.
390/390 [==============================] - 19s 48ms/step - loss: 0.3814 - acc: 0.8664 - val_loss: 0.5397 - val_acc: 0.8199
Epoch 37/50

Epoch 00037: LearningRateScheduler setting learning rate to 0.0040051266.
390/390 [==============================] - 19s 48ms/step - loss: 0.3791 - acc: 0.8653 - val_loss: 0.5565 - val_acc: 0.8208
Epoch 38/50

Epoch 00038: LearningRateScheduler setting learning rate to 0.0039053347.
390/390 [==============================] - 19s 48ms/step - loss: 0.3727 - acc: 0.8693 - val_loss: 0.5308 - val_acc: 0.8282
Epoch 39/50

Epoch 00039: LearningRateScheduler setting learning rate to 0.0038103948.
390/390 [==============================] - 19s 48ms/step - loss: 0.3696 - acc: 0.8688 - val_loss: 0.5364 - val_acc: 0.8258
Epoch 40/50

Epoch 00040: LearningRateScheduler setting learning rate to 0.0037199613.
390/390 [==============================] - 19s 48ms/step - loss: 0.3634 - acc: 0.8703 - val_loss: 0.5676 - val_acc: 0.8224
Epoch 41/50

Epoch 00041: LearningRateScheduler setting learning rate to 0.0036337209.
390/390 [==============================] - 19s 48ms/step - loss: 0.3648 - acc: 0.8691 - val_loss: 0.5436 - val_acc: 0.8277
Epoch 42/50

Epoch 00042: LearningRateScheduler setting learning rate to 0.0035513886.
390/390 [==============================] - 19s 48ms/step - loss: 0.3674 - acc: 0.8699 - val_loss: 0.5345 - val_acc: 0.8244
Epoch 43/50

Epoch 00043: LearningRateScheduler setting learning rate to 0.0034727045.
390/390 [==============================] - 19s 48ms/step - loss: 0.3620 - acc: 0.8722 - val_loss: 0.5551 - val_acc: 0.8231
Epoch 44/50

Epoch 00044: LearningRateScheduler setting learning rate to 0.0033974315.
390/390 [==============================] - 19s 48ms/step - loss: 0.3583 - acc: 0.8737 - val_loss: 0.5502 - val_acc: 0.8196
Epoch 45/50

Epoch 00045: LearningRateScheduler setting learning rate to 0.0033253525.
390/390 [==============================] - 19s 48ms/step - loss: 0.3588 - acc: 0.8729 - val_loss: 0.5189 - val_acc: 0.8311
Epoch 46/50

Epoch 00046: LearningRateScheduler setting learning rate to 0.0032562683.
390/390 [==============================] - 19s 48ms/step - loss: 0.3519 - acc: 0.8752 - val_loss: 0.5521 - val_acc: 0.8225
Epoch 47/50

Epoch 00047: LearningRateScheduler setting learning rate to 0.0031899962.
390/390 [==============================] - 19s 48ms/step - loss: 0.3553 - acc: 0.8732 - val_loss: 0.5481 - val_acc: 0.8256
Epoch 48/50

Epoch 00048: LearningRateScheduler setting learning rate to 0.0031263678.
390/390 [==============================] - 19s 49ms/step - loss: 0.3488 - acc: 0.8763 - val_loss: 0.5294 - val_acc: 0.8299
Epoch 49/50

Epoch 00049: LearningRateScheduler setting learning rate to 0.0030652281.
390/390 [==============================] - 19s 49ms/step - loss: 0.3497 - acc: 0.8758 - val_loss: 0.5398 - val_acc: 0.8294
Epoch 50/50

Epoch 00050: LearningRateScheduler setting learning rate to 0.0030064338.
390/390 [==============================] - 19s 48ms/step - loss: 0.3475 - acc: 0.8772 - val_loss: 0.5549 - val_acc: 0.8220
Model took 946.99 seconds to train
