### Cdiscount Image Classification Challenge

Cdiscount Image Classification Challange Ongoing Solution.

## Results

Right now getting top1 accuray - 0.64 accuracy & top5 accuracy - 0.81

Will provide the weights once challenge is over.

Using an SE-Denset161 model, using intitial iterations with data augmention & last few iterations without any augmentation

use adam_accumulate or sgd_accumulate for batch accumulation, code also provide in the keras issue,
https://github.com/fchollet/keras/issues/3556


Learning rate 0-3 = 0.1
			  4-7 = 0.001
			  8-10 = 0.0001