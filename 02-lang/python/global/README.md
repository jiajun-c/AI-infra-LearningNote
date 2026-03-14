# Python 全局对象

Python实际上没有真正的跨文件全局变量，其可见仅限于当前的py文件

Python的全局变量也分为可变对象和不可变对象

不可变对象有`int,str,tuple`，需要先用global修改变量的指向再进行修改

对于可变对象如`list,dict`，由于其修改的是对象内部xian,所以不需要global修改变量的指向

