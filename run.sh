# 8 gpus for imagenet
# 2 gpus for cifar, speech_commands
# 1 gpu  for covtype, higgs

#############
# table 1,4 #
#############

python main_pretext.py 'data' --dataset cifar10  -a resnet50 --cos --warm --lincls --tb --resume true --method npair --imix none   --proj mlpbn  --temp 0.2 --epochs 2000 --trial 0 --multiprocessing-distributed --dist-url 'tcp://localhost:10001'
python main_pretext.py 'data' --dataset cifar10  -a resnet50 --cos --warm --lincls --tb --resume true --method npair --imix imixup --proj mlpbn  --temp 0.2 --epochs 4000 --trial 0 --multiprocessing-distributed --dist-url 'tcp://localhost:10002'
python main_pretext.py 'data' --dataset cifar10  -a resnet50 --cos --warm --lincls --tb --resume true --method moco  --imix none   --proj mlpbn1 --temp 0.2 --epochs 1000 --trial 0 --multiprocessing-distributed --dist-url 'tcp://localhost:10003' --lincls-more
python main_pretext.py 'data' --dataset cifar10  -a resnet50 --cos --warm --lincls --tb --resume true --method moco  --imix imixup --proj mlpbn1 --temp 0.2 --epochs 4000 --trial 0 --multiprocessing-distributed --dist-url 'tcp://localhost:10004' --lincls-more
python main_pretext.py 'data' --dataset cifar10  -a resnet50 --cos --warm --lincls --tb --resume true --method byol  --imix none   --proj mlpbn1 --pred mlpbn1 --epochs 1000 --trial 0 --multiprocessing-distributed --dist-url 'tcp://localhost:10005'
python main_pretext.py 'data' --dataset cifar10  -a resnet50 --cos --warm --lincls --tb --resume true --method byol  --imix imixup --proj mlpbn1 --pred mlpbn1 --epochs 2000 --trial 0 --multiprocessing-distributed --dist-url 'tcp://localhost:10016'

python main_pretext.py 'data' --dataset cifar100 -a resnet50 --cos --warm --lincls --tb --resume true --method npair --imix none   --proj mlpbn  --temp 0.2 --epochs 4000 --trial 0 --multiprocessing-distributed --dist-url 'tcp://localhost:10011'
python main_pretext.py 'data' --dataset cifar100 -a resnet50 --cos --warm --lincls --tb --resume true --method npair --imix imixup --proj mlpbn  --temp 0.2 --epochs 2000 --trial 0 --multiprocessing-distributed --dist-url 'tcp://localhost:10012'
python main_pretext.py 'data' --dataset cifar100 -a resnet50 --cos --warm --lincls --tb --resume true --method moco  --imix none   --proj mlpbn1 --temp 0.2 --epochs 1000 --trial 0 --multiprocessing-distributed --dist-url 'tcp://localhost:10013' --lincls-more
python main_pretext.py 'data' --dataset cifar100 -a resnet50 --cos --warm --lincls --tb --resume true --method moco  --imix imixup --proj mlpbn1 --temp 0.2 --epochs 2000 --trial 0 --multiprocessing-distributed --dist-url 'tcp://localhost:10014' --lincls-more
python main_pretext.py 'data' --dataset cifar100 -a resnet50 --cos --warm --lincls --tb --resume true --method byol  --imix none   --proj mlpbn1 --pred mlpbn1 --epochs 1000 --trial 0 --multiprocessing-distributed --dist-url 'tcp://localhost:10015'
python main_pretext.py 'data' --dataset cifar100 -a resnet50 --cos --warm --lincls --tb --resume true --method byol  --imix imixup --proj mlpbn1 --pred mlpbn1 --epochs 4000 --trial 0 --multiprocessing-distributed --dist-url 'tcp://localhost:10016'

python main_pretext.py 'data' --dataset speech_commands -a resnet50 --warm --lincls --tb --resume true --method npair --imix none   --proj mlpbn --temp 0.5 --epochs 500 --schedule 300 400 --trial 0 --multiprocessing-distributed --dist-url 'tcp://localhost:10021' -j 16
python main_pretext.py 'data' --dataset speech_commands -a resnet50 --warm --lincls --tb --resume true --method npair --imix imixup --proj mlpbn --temp 0.5 --epochs 500 --schedule 300 400 --trial 0 --multiprocessing-distributed --dist-url 'tcp://localhost:10022' -j 16
python main_pretext.py 'data' --dataset speech_commands -a resnet50 --warm --lincls --tb --resume true --method moco  --imix none   --proj mlpbn --temp 0.5 --epochs 500 --schedule 300 400 --trial 0 --multiprocessing-distributed --dist-url 'tcp://localhost:10023' -j 16
python main_pretext.py 'data' --dataset speech_commands -a resnet50 --warm --lincls --tb --resume true --method moco  --imix imixup --proj mlpbn --temp 0.5 --epochs 500 --schedule 300 400 --trial 0 --multiprocessing-distributed --dist-url 'tcp://localhost:10024' -j 16
python main_pretext.py 'data' --dataset speech_commands -a resnet50 --warm --lincls --tb --resume true --method byol  --imix none   --proj mlpbn --pred mlpbn1 --epochs 500 --schedule 300 400 --trial 0 --multiprocessing-distributed --dist-url 'tcp://localhost:10025' -j 16
python main_pretext.py 'data' --dataset speech_commands -a resnet50 --warm --lincls --tb --resume true --method byol  --imix imixup --proj mlpbn --pred mlpbn1 --epochs 500 --schedule 300 400 --trial 0 --multiprocessing-distributed --dist-url 'tcp://localhost:10026' -j 16

python main_pretext.py 'data' --dataset covtype -a mlp_5layer --warm --cos --lincls --tb --resume true --method npair --imix none   --proj mlpbn --temp 0.1 --epochs 500 --trial 0 -b 512 --alpha 0.0 --inputdrop 0.2
python main_pretext.py 'data' --dataset covtype -a mlp_5layer --warm --cos --lincls --tb --resume true --method npair --imix imixup --proj mlpbn --temp 0.1 --epochs 500 --trial 0 -b 512 --alpha 2.0 --inputdrop 0.2
python main_pretext.py 'data' --dataset covtype -a mlp_5layer --warm --cos --lincls --tb --resume true --method moco  --imix none   --proj mlpbn --temp 0.1 --epochs 500 --trial 0 -b 512 --alpha 0.0 --inputdrop 0.2
python main_pretext.py 'data' --dataset covtype -a mlp_5layer --warm --cos --lincls --tb --resume true --method moco  --imix imixup --proj mlpbn --temp 0.1 --epochs 500 --trial 0 -b 512 --alpha 2.0 --inputdrop 0.0
python main_pretext.py 'data' --dataset covtype -a mlp_5layer --warm --cos --lincls --tb --resume true --method byol  --imix none   --proj mlpbn --pred mlpbn --epochs 500 --trial 0 -b 512 --alpha 0.0 --inputdrop 0.2
python main_pretext.py 'data' --dataset covtype -a mlp_5layer --warm --cos --lincls --tb --resume true --method byol  --imix imixup --proj mlpbn --pred mlpbn --epochs 500 --trial 0 -b 512 --alpha 2.0 --inputdrop 0.0


###########
# table 2 #
###########

python main_pretext.py 'data' --dataset imagenet -a resnet50 --cos --warm --lincls --tb --resume true --method moco  --imix none   --proj mlp --temp 0.2 --epochs 800 --trial 0 --multiprocessing-distributed --dist-url 'tcp://localhost:10031' --lr 0.03 -b 512 --qlen 65536 --alpha 0.0 --class-ratio 0.1
python main_pretext.py 'data' --dataset imagenet -a resnet50 --cos --warm --lincls --tb --resume true --method moco  --imix imixup --proj mlp --temp 0.2 --epochs 800 --trial 0 --multiprocessing-distributed --dist-url 'tcp://localhost:10032' --lr 0.03 -b 512 --qlen 65536 --alpha 0.2 --class-ratio 0.1

python main_pretext.py 'data' --dataset imagenet -a resnet50 --cos --warm --lincls --tb --resume true --method moco  --imix none   --proj mlp --temp 0.2 --epochs 800 --trial 0 --multiprocessing-distributed --dist-url 'tcp://localhost:10033' --lr 0.03 -b 512 --qlen 65536 --alpha 0.0
python main_pretext.py 'data' --dataset imagenet -a resnet50 --cos --warm --lincls --tb --resume true --method moco  --imix imixup --proj mlp --temp 0.2 --epochs 800 --trial 0 --multiprocessing-distributed --dist-url 'tcp://localhost:10034' --lr 0.03 -b 512 --qlen 65536 --alpha 0.2

python main_pretext.py 'data' --dataset higgs100Kall -a mlp_5layer --warm --cos --pinv --tb --resume true --method moco  --imix none   --proj mlpbn --temp 0.1 --epochs 500 --trial 0 -b 512 --alpha 0.0 --inputdrop 0.2
python main_pretext.py 'data' --dataset higgs100Kall -a mlp_5layer --warm --cos --pinv --tb --resume true --method moco  --imix imixup --proj mlpbn --temp 0.1 --epochs 500 --trial 0 -b 512 --alpha 1.0 --inputdrop 0.2

python main_pretext.py 'data' --dataset higgs1Mall   -a mlp_5layer --warm --cos --pinv --tb --resume true --method moco  --imix none   --proj mlpbn --temp 0.1 --epochs 500 --trial 0 -b 512 --alpha 0.0 --inputdrop 0.2
python main_pretext.py 'data' --dataset higgs1Mall   -a mlp_5layer --warm --cos --pinv --tb --resume true --method moco  --imix imixup --proj mlpbn --temp 0.1 --epochs 500 --trial 0 -b 512 --alpha 1.0 --inputdrop 0.2


###########
# table 3 #
###########

python main_pretext.py 'data' --dataset cifar10  -a resnet50 --cos --warm --lincls --tb --resume true --method moco  --imix none    --proj mlpbn1 --temp 0.2 --epochs 500  --trial 0 --multiprocessing-distributed --dist-url 'tcp://localhost:10101' --no-aug
python main_pretext.py 'data' --dataset cifar10  -a resnet50 --cos --warm --lincls --tb --resume true --method moco  --imix imixup  --proj mlpbn1 --temp 0.2 --epochs 1000 --trial 0 --multiprocessing-distributed --dist-url 'tcp://localhost:10102' --no-aug --inputmix 2

python main_pretext.py 'data' --dataset cifar100 -a resnet50 --cos --warm --lincls --tb --resume true --method moco  --imix none    --proj mlpbn1 --temp 0.2 --epochs 1000 --trial 0 --multiprocessing-distributed --dist-url 'tcp://localhost:10111' --no-aug
python main_pretext.py 'data' --dataset cifar100 -a resnet50 --cos --warm --lincls --tb --resume true --method moco  --imix imixup  --proj mlpbn1 --temp 0.2 --epochs 500  --trial 0 --multiprocessing-distributed --dist-url 'tcp://localhost:10112' --no-aug --inputmix 2

python main_pretext.py 'data' --dataset speech_commands -a resnet50 --warm --lincls --tb --resume true --method moco  --imix none   --proj mlpbn --temp 0.2 --epochs 500 --schedule 300 400 --trial 0 --multiprocessing-distributed --dist-url 'tcp://localhost:10121' -j 16 --no-aug
python main_pretext.py 'data' --dataset speech_commands -a resnet50 --warm --lincls --tb --resume true --method moco  --imix imixup --proj mlpbn --temp 0.2 --epochs 500 --schedule 300 400 --trial 0 --multiprocessing-distributed --dist-url 'tcp://localhost:10122' -j 16 --no-aug

python main_pretext.py 'data' --dataset covtype -a mlp_5layer --warm --cos --lincls --tb --resume true --method moco  --imix none   --proj mlpbn --temp 0.1 --epochs 500 --trial 0 -b 512 --alpha 0.0 --inputdrop 0.0
python main_pretext.py 'data' --dataset covtype -a mlp_5layer --warm --cos --lincls --tb --resume true --method moco  --imix imixup --proj mlpbn --temp 0.1 --epochs 500 --trial 0 -b 512 --alpha 2.0 --inputdrop 0.0

python main_pretext.py 'data' --dataset higgs100Kall -a mlp_5layer --warm --cos --lincls --tb --resume true --method moco  --imix none   --proj mlpbn --temp 0.1 --epochs 500 --trial 0 -b 512 --alpha 0.0 --inputdrop 0.0
python main_pretext.py 'data' --dataset higgs100Kall -a mlp_5layer --warm --cos --lincls --tb --resume true --method moco  --imix imixup --proj mlpbn --temp 0.1 --epochs 500 --trial 0 -b 512 --alpha 1.0 --inputdrop 0.0

python main_pretext.py 'data' --dataset higgs1Mall   -a mlp_5layer --warm --cos --lincls --tb --resume true --method moco  --imix none   --proj mlpbn --temp 0.1 --epochs 500 --trial 0 -b 512 --alpha 0.0 --inputdrop 0.0
python main_pretext.py 'data' --dataset higgs1Mall   -a mlp_5layer --warm --cos --lincls --tb --resume true --method moco  --imix imixup --proj mlpbn --temp 0.1 --epochs 500 --trial 0 -b 512 --alpha 1.0 --inputdrop 0.0

