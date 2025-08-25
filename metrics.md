    all_columns
{'MSE': np.float64(376547.981654314), 'RMSE': np.float64(613.6350557573402), 'MAE': np.float64(459.2782207392964), 'MAPE': np.float64(564.0313000405498), 'R²': np.float64(0.18786142196737066), 'EVS': np.float64(0.30074575482928045)}


araujo_columns
{'MSE': np.float64(339658.61762409436), 'RMSE': np.float64(582.8023829945228), 'MAE': np.float64(466.83642472399634), 'MAPE': np.float64(277.8216099561843), 'R²': np.float64(0.1158999612905427), 'EVS': np.float64(0.17223482235432608)}


utilizando StandardScaler
{'MSE': np.float64(263969.01230389834), 'RMSE': np.float64(513.7791474008052), 'MAE': np.float64(352.54297579677996), 'MAPE': np.float64(207.04115270199824), 'R²': np.float64(0.5085071912424666), 'EVS': np.float64(0.5540702075408006)}

standardscaler - corte nas horas
{'MSE': np.float64(251599.9703793581), 'RMSE': np.float64(501.5974186330688), 'MAE': np.float64(389.35308715598853), 'MAPE': np.float64(46.19406468668011), 'R²': np.float64(0.31884286158212594), 'EVS': np.float64(0.35178381318822594)}


minmax - corte nas horas
{'MSE': np.float64(250135.25926071318), 'RMSE': np.float64(500.13524097059303), 'MAE': np.float64(392.8558913435922), 'MAPE': np.float64(42.14758838434525), 'R²': np.float64(0.19524291651213388), 'EVS': np.float64(0.20865695759970782)}


minmax - 64 neuronios
{'MSE': np.float64(260463.8922012173), 'RMSE': np.float64(510.3566323672274), 'MAE': np.float64(398.5617537582022), 'MAPE': np.float64(29.978548103718573), 'R²': np.float64(0.09667117002113546), 'EVS': np.float64(0.1023628475322057)}


standard - 64 neuronois
{'MSE': np.float64(251905.93647804277), 'RMSE': np.float64(501.9023176655421), 'MAE': np.float64(388.8281146817883), 'MAPE': np.float64(41.521868548577615), 'R²': np.float64(0.3047123868082834), 'EVS': np.float64(0.34239215676671986)}


standard - 64 neuronios alteração na hora 6 as 18
{'MSE': np.float64(265130.0676029108), 'RMSE': np.float64(514.9078243753058), 'MAE': np.float64(358.5298852310504), 'MAPE': np.float64(128.07526431551176), 'R²': np.float64(0.6615775715322194), 'EVS': np.float64(0.6893885696815952)}

standard - 64 neuronios, 32 na segunda camada,
'MSE': np.float64(260576.22039591728), 'RMSE': np.float64(510.46666923112355), 'MAE': np.float64(356.576201113789), 'MAPE': np.float64(166.15240454256937), 'R²': np.float64(0.677646064849025), 'EVS': np.float64(0.7052395725808998)

standard - 64 neuronios sigmoid, 32 relu na segunda camada
{'MSE': np.float64(292397.48919782956), 'RMSE': np.float64(540.7379117445249), 'MAE': np.float64(397.3566344903151), 'MAPE': np.float64(208.84866280673828), 'R²': np.float64(0.637543592936437), 'EVS': np.float64(0.6728230139531708)}


standard - 32 sigmoid
{'MSE': np.float64(350250.23133491713), 'RMSE': np.float64(591.8194246008803), 'MAE': np.float64(497.0628809226639), 'MAPE': np.float64(130.38297202483713), 'R²': np.float64(0.5227701355717758), 'EVS': np.float64(0.5462156665736178)}

standard - 64 sigmoid
{'MSE': np.float64(353394.6212173173), 'RMSE': np.float64(594.4700339103034), 'MAE': np.float64(497.2255272393899), 'MAPE': np.float64(107.76901990921858), 'R²': np.float64(0.5709977918738139), 'EVS': np.float64(0.59061457810907)}

standard - 64 sigmoid early stopping
{'MSE': np.float64(2374691.9683876694), 'RMSE': np.float64(1541.0035588497742), 'MAE': np.float64(1303.8764610419757), 'MAPE': np.float64(60.616598867525184), 'R²': np.float64(-52.90802999997271), 'EVS': np.float64(-30.332596234205926)}    


standard - 32 relu early stopping
{'MSE': np.float64(1053528.0748567968), 'RMSE': np.float64(1026.4151571643888), 'MAE': np.float64(925.6080203226271), 'MAPE': np.float64(67.8527329334959), 'R²': np.float64(-24.294513503626167), 'EVS': np.float64(-22.22768679193837)}