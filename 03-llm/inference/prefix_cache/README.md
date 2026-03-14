# Prefix caching

Prefix caching是为了解决当一些输入他们有共同的prompt的时候，可以让共同的前缀部分只缓存一份

假设输入为 You like -> math 和 You like -> English，当block size为2的时候，他们将会共享前面的You like

