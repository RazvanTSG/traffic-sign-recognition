Approach 1: Similar to the CS50 Course

Stats: accuracy: 0.9662 - loss: 0.1318
4ms/Epoch step; 1ms/step at predicting ( 4 / 1 timp )

Approach 2: Similar to CS50 approach, but making it more complex ( more layers, more filters):

2.1: Incercam more filters:
accuracy: 0.9651 - loss: 0.1527
8ms/Epoch step; 2ms/step at predicting

2.2 Adaugam Inca un layer, tot cu 32 filters, nu dam pool inca o data:
accuracy: 0.9852 - loss: 0.0596
4 / 1 timp

2.3 Adaugam Inca un layer, tot cu 32 filters, dam pool
accuracy: 0.9768 - loss: 0.0898
3 / 2 ms

2.4 La ultimul layer, ii punem 64 filters, pt detalii mai amanuntite la al doilea layer, nu dam pool
accuracy: 0.9845 - loss: 0.0677
6.5 / 2 ms

2.5 La ultimul layer, ii punem 64 filters, pt detalii mai amanuntite, dam pool dupa
accuracy: 0.9885 - loss: 0.053
4 / 2 ms

2.6 Same ca la 2.5, dar cu dropout de 0.3 in loc de 0.5
accuracy: 0.9854 - loss: 0.0621
4 / 2 ms

2.7 Cu 0.4 Dropout:
accuracy: 0.9886 - loss: 0.0511

2.8 Incercam Augmentation Layers ( rotim putin semnele )
accuracy: 0.9529 - loss: 0.1777
6 / 2 ms

2.9 Incercam primul Kernel cu (5,5) in loc de (3,3)                  ---> The best
accuracy: 0.9910 - loss: 0.0414
4 / 2

2.10 (6,6)
accuracy: 0.9882 - loss: 0.0545
4 / 2

2.11 (5,5) si (5,5) la al doilea layer
accuracy: 0.9879 - loss: 0.0604
4 / 2



