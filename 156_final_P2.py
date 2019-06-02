# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '156.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import all_in_one_P2 as Data
# ccc=0

# li_temp=['0']

# # model=[]

# for a in model:
#     if type(a)==str: litemp_append(str(ccc))
#     else:  li_temp.append(int(ccc))

#     ccc+=1


li=['0']
dic={0:'geisel_library.png'}

# li=['0','1','2',3,'4','5',6,7,'8','9']
# dic={0:'geisel_library.png',3:'3.png',6:'6.png',7:'7.png'}

class Ui_Dialog(object):
    t=0
    def __init__(self):
        Ui_Dialog.t =0
        self.Model=Data.Model()
        

    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1251, 668)

        self.gridLayout_2 = QtWidgets.QGridLayout(Dialog)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.input = QtWidgets.QLineEdit(Dialog)
        self.input.setText("")
        self.input.setObjectName("input")

        self.gridLayout_2.addWidget(self.input, 0, 0, 1, 1)

        self.search = QtWidgets.QPushButton(Dialog)
        self.search.setObjectName("search")

        self.gridLayout_2.addWidget(self.search, 0, 1, 1, 1)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")

        self.prev = QtWidgets.QPushButton(Dialog)
        self.prev.setObjectName("prev")

        self.gridLayout.addWidget(self.prev, 1, 1, 1, 1)

        self.page_number = QtWidgets.QLineEdit(Dialog)
        self.page_number.setObjectName("page_number")
        self.page_number.setAlignment(QtCore.Qt.AlignCenter)

        self.gridLayout.addWidget(self.page_number, 1, 3, 1, 1)

        spacerItem = QtWidgets.QSpacerItem(48, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

        self.gridLayout.addItem(spacerItem, 1, 4, 1, 1)

        self.next = QtWidgets.QPushButton(Dialog)
        self.next.setObjectName("next")
        self.gridLayout.addWidget(self.next, 1, 5, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(118, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 1, 6, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(128, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem2, 1, 0, 1, 1)
        self.content = QtWidgets.QPlainTextEdit(Dialog)
        self.content.setPlainText("")
        self.content.setObjectName("content")
        self.gridLayout.addWidget(self.content, 0, 0, 1, 7)
        spacerItem3 = QtWidgets.QSpacerItem(58, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem3, 1, 2, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 1, 0, 1, 2)


        self.f=self.content.font()
        self.f.setPointSize(33) # sets the size to 30
        self.content.setFont(self.f)

        #self.label.setAlignment(Qt.AlignCenter)
        self.page_number.setText('0')   #set function on text.
        # self.page_number.text() #get function from text.
        self.content.hide()


        self.pic = QtWidgets.QLabel(Dialog)
        
        self.pic.setObjectName("pic")
        self.gridLayout.addWidget(self.pic, 0, 0, 1, 7)



        # self.pic = QtWidgets.QLabel(Dialog)
        # self.pic.setGeometry(QtCore.QRect(40, 150, 691, 321))
        # self.pic.setObjectName("pic")


        # self.pixmap = QtGui.QPixmap('geisel_library.png')
        # self.pixmap = self.pixmap.scaled(self.pic.width(), self.pic.height())
        # self.pic.setPixmap(self.pixmap)
        # self.pic.setMinimumSize(1, 1)
        # self.pic.show()

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        self.search.clicked.connect(self.Search)
        self.next.clicked.connect(self.Next)
        self.prev.clicked.connect(self.Prev)
        # self.Picture0()

    def Picture0(self):
        self.pixmap = QtGui.QPixmap(dic[0])
        self.pixmap = self.pixmap.scaled(self.pic.width(), self.pic.height())
        self.pic.setPixmap(self.pixmap)
        self.pic.setMinimumSize(1, 1)
        self.pic.show()

    def Picture(self):
        self.pixmap = QtGui.QPixmap(self.dic[int(self.page_number.text())])
        self.pixmap = self.pixmap.scaled(self.pic.width(), self.pic.height())
        self.pic.setPixmap(self.pixmap)
        self.pic.setMinimumSize(1, 1)
        self.pic.show()




    def update_sentence(self,str1):
        
        self.li,self.dic=self.Model.pipeline(str1)
        # li1=predict(str1)   #later change
        # return li

    def Search(self):

        str1= self.input.text()
        if len(str1) == 0:return 
        print(str1)

        # li=update_sentence(str1)
        self.update_sentence(str1)      #check it later

        self.page_number.setText('1')
        c=1
        
        if type(self.li[c]) == str:
            self.pic.hide()
            self.content.show()
            self.content.setPlainText(self.li[c]) 
            # self.t=0
        else:
            self.content.hide()
            # self.t=1
            self.Picture()
            

    def Next(self):
        str1= self.input.text()
        if len(str1) == 0:return 
        c=int(self.page_number.text())+1
        if c== len(self.li): return
        self.page_number.setText(str(c))
        if type(self.li[c]) == str:
            self.pic.hide()
            self.content.show()
            self.content.setPlainText(self.li[c])
        else:
            self.content.hide()
            # self.t+=1
            self.Picture()

            

    def Prev(self):
        str1= self.input.text()
        if len(str1) == 0:return 
        c=int(self.page_number.text())
        if c <= 1: return
        c-=1
        self.page_number.setText(str(c))      
        if type(self.li[c]) == str:
            self.pic.hide()
            self.content.show()
            self.content.setPlainText(self.li[c])
        else:
            self.content.hide()
            # self.t-=1
            self.Picture()


    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.input.setText(_translate("Dialog", ""))
        self.search.setText(_translate("Dialog", "Search"))
        self.content.setPlainText(_translate("Dialog", "content\n"
""))
        self.prev.setText(_translate("Dialog", "< Prev"))
        self.next.setText(_translate("Dialog", "Next >"))


if __name__ == "__main__":
    import sys
    
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    ui.Picture0()
    sys.exit(app.exec_())

