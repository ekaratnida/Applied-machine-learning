## Reference

Overfitting ภาพลวงตาของระบบการลงทุน: https://www.siamquant.com/illusion-of-overfitting/?fbclid=IwAR0jSMSHURwhmJ3Y1XYIcmtAsTFqyBa6ihcuC7j1uZSCEQntbPiHiN4X0pQ

How ridge and lasso work?: 
- https://www.youtube.com/watch?v=Xm2C_gTAl8c
- https://www.linkedin.com/pulse/tutorial-ridge-lasso-regression-subhajit-mondal/?trk=read_related_article-card_title

Explain: https://towardsdatascience.com/regularization-in-machine-learning-connecting-the-dots-c6e030bfaddd

## Note
The parameter l1_ratio corresponds to alpha in the glmnet R package while alpha corresponds to the lambda parameter in glmnet. Specifically, l1_ratio = 1 is the lasso penalty. Currently, l1_ratio <= 0.01 is not reliable, unless you supply your own sequence of alpha.

## Ridge VS Lasso
![lasso](https://user-images.githubusercontent.com/69342162/217750172-e885491a-9373-436f-9eb6-0ac81fb22c33.gif)

