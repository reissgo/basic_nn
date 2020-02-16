from tkinter import *

from basic_nn import *

def run_function():

    global x_train, x_test, y_train, y_test
    x_train, x_test, y_train, y_test = generate_dataset(1000,.3)

    print(f"x_train=\n{x_train} \ny_train=\n{y_train}")

    # build model 2 -> 5 -> 1

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(int(numhid_entry_dodad.get()),input_dim = 2, activation="sigmoid"),
        tf.keras.layers.Dense(1, activation="sigmoid")
        ])

    # compile model

    # MSE = min squared errot
    optimizer_to_use = tf.keras.optimizers.SGD(learning_rate=0.1)
    model.compile(optimizer=optimizer_to_use, loss="MSE")

    #train

    model.fit(x_train,y_train,epochs=int(epochs_entry_dodad.get()))

    # evaluate

    print("Evaluate...")
    model.evaluate(x_test,y_test,verbose=0)

    # make prediction

    print("predictions...")
    input_data_for_predictions = [[.1, .1], [.1, .3], [.4, .4], [.2, .6]]

    predictions = model.predict(input_data_for_predictions)

    for ind,out in zip(input_data_for_predictions, predictions):
        print(f"{ind[0]} + {ind[1]} = {out[0]}")

    print(predictions)



root = Tk()
root.title("Basic neural network")
row_ctr = 0
pad = 5

epochs_label = Label(root,text="Epochs")
epochs_label.grid(row=row_ctr, column=1,sticky=E)
epochs_entry_dodad = Entry(root)
epochs_entry_dodad.insert(0, "100")
epochs_entry_dodad.grid(row=row_ctr, column=2, padx=pad, pady=pad)
row_ctr+=1

numhid_label = Label(root,text="num nodes in hidden layer")
numhid_label.grid(row=row_ctr, column=1,sticky=E)
numhid_entry_dodad = Entry(root)
numhid_entry_dodad.insert(0, "5")
numhid_entry_dodad.grid(row=row_ctr, column=2, padx=pad, pady=pad)
row_ctr+=1

frame = Frame(root)
frame.grid(row=row_ctr,column=1,columnspan=2)
runbut = Button(frame, text="Run!", command = run_function)
runbut.grid(row=row_ctr, column=1, padx=pad, pady=pad)
row_ctr+=1

root.mainloop()