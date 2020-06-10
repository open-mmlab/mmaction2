![parrots](imgs/parrots_logo.png)

-----------------------------------

Parrots is a deep learning framework independently developed by the platform group. 

- Compatible with Pytorch interface
- Support faster training speed
- Support linear expansion (128 GPUS acceleration ratio is over 90%)

The goal of Parrots is supporting:

- 1000+ GPUS parallel training
- A hundred billion level parametric model
- Billions of training samples
- Ten million category sorting tasks

## Advantage

Parrots can combine [AutoLink](http://autolink.parrots.sensetime.com) and [Pavi](http://pavi.parrots.sensetime.com) for a best model training experience. 

Parrots, AutoLink and Pavi can open up the entire process from production to deployment, make the task easier and faster.

Benchmark:

- to do

## Install

Reference: [https://confluence.sensetime.com/pages/viewpage.action?pageId=131288226](https://confluence.sensetime.com/pages/viewpage.action?pageId=131288226)

## Pytorch to Parrots

#### Reference

[http://parrots.sensetime.com](http://parrots.sensetime.com)

The documentation includes the usage of the parrots basic interface (most of which is the same as the pytorch interface), the precision analysis tool usage, the model transformation tool, and the Timeline profile tool.

#### Operator

Parrots currently supports most pytorch interfaces, and most of the time the pytorch code runs directly by parrots.



## Model Convert (Parrots to Caffe)

Support convert model from parrots to caffe.

|model|support|
|---|---|
|TSM2D|yes|
|SlowFast3D|yes|
|R2Plus1D|yes|
|TSN2D|yes|
|TIN2D|yes|
|CSN3D|yes|
|BSNTEM|yes|
|BSNPEM|yes|

note: BSNTEM and BSNPEM only support convert `_forward()`.

#### How to use

Use environment variable `MMACTION_MODEL_CONVERT=1` to control execution training code or model conversion code.

1) code prepare

```
git clone git@gitlab.sz.sensetime.com:open-mmlab/mmaction-lite.git
git clone git@gitlab.bj.sensetime.com:platform/ParrotsDL/parrots.convert.git
```
contact `luopeichao@sensetime.com` for `parrots.convert` permission.

2) install mmaction and parrots.convert

```
cd mmaction-lite
pip install -e .
```
and
```
cd parrots.convert
pip install -e .
```

3) enter work directory

```
cd parrots.convert/tools/mmaction
```

4) execute ./run.sh

```
./run.sh Platform --model=SlowFast3D --config=./tests/slowfast/slowfast_32x2x1_r50_3d_kinetics400_256e.py --checkpoint=./tests/slowfast/epoch.pth --mergebn --savedir=caffe_model_slowfast --inputsize="6,3,32,224,224"
```

the args of **./run.sh**, first is cluster partition, all the following are python args.

The python args are illustrated below.

|args|description|other|
|---|---|---|
|--model|The name of the model that needs to be converted|required|
|--config|path of the config|required|
|--checkpoint|path of the checkpoint|required|
|--batchsize|batchsize of the input data|optional, default:8|
|--mergebn|merger bn layer|optional, default:false|
|--savedir|the directory where the transformed model is saved|optional, default:caffe_model|
|--inputsize|size of the input data, for example, "8,8,3,224,224"|required|
--saveinput|whether to save the generated input data|optional, default:false|

## Model Convert (Parrots to Pytorch)

Reference: [https://confluence.sensetime.com/pages/viewpage.action?pageId=72310284](https://confluence.sensetime.com/pages/viewpage.action?pageId=72310284)

## Other support

Contact: liujie3@sensetime.com

Home page: [https://confluence.sensetime.com/display/PlatformSupport/Parrots](https://confluence.sensetime.com/display/PlatformSupport/Parrots)

