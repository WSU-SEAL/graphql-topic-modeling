require("ggplot2")
mydata = read.csv(file.choose(), header = TRUE)

ggplot(data=mydata, aes(x= NotAnswered, y= View, size=percentQuestions,  color= percentQuestions)) +
  xlab("% questions w/o accepted answers") +
  ylab("Median views") +
  labs(color = "Topic Size") +
  scale_size(range = c(5, 20)) +
  geom_point(alpha=1, show.legend = FALSE) +
  # theme(legend.position = "none") +
  scale_colour_gradientn(colours=rainbow(3)) +
  geom_vline(color="red", xintercept = 0.627, show.legend = FALSE) +
  geom_hline(color="red", yintercept = 387, show.legend = FALSE) +
  geom_text(angle=20, size=3, color="darkblue", data = mydata, mapping = aes(x = NotAnswered, y = View,  label=TopicName)) +
  geom_label( label = "Unpopular & Easy",    x = 0.58, y = 330, size = 4, colour = "blue" ) +
  geom_label(  label = "Unpopular & Difficult",   x = 0.69, y = 340, size = 4, colour = "red" )+
  geom_label(    label = "Popular & Easy",   x = 0.58, y = 455, size = 4, colour = "darkgreen" )+
  geom_label(  label = "Popular & Difficult",   x = 0.69, y = 440, size = 4, colour = "brown" )+ 
  xlim(0.54,0.7) +
#geom_label(size=3, color="red", show.legend= FALSE, x = 0.58, y = 320,  label="Popular & Easy") 
  
  #geom_text(size=3.6, color="white", data = mydata, show.legend= FALSE, mapping = aes(x = Difficulty, y = Popularity, label=percent(percentQuestions, accuracy = 0.1), fontface="bold")) +
  geom_label(size=3, color="red", show.legend= FALSE, x = 0.58, y = 320,  label="Popular & Easy") +
  geom_label(size=3, color="red", show.legend= FALSE, x = 0.45, y = 0.6,  label="Unpopular & Easy") +
  geom_label(size=3, color="red", show.legend= FALSE, x = 0.75, y = 320,  label="Unpopular & Difficult") +
  geom_label(size=3, color="red", show.legend= FALSE, x = 0.75, y = 450,  label="Popular & Difficult")
#+
  #guides(size = FALSE) 