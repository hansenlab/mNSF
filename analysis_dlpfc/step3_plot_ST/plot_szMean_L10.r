
### load pacakges 
library(tidyverse)
library(ggplot2)
library(Matrix)
# library(Rmisc)#
library(ggforce)#
# library(rjson)#
library(cowplot)
library(RColorBrewer)
library(grid)
require(nnet)
# require(nnet)

group.colors <- c(Layer1 = "#FC8D62", Layer2 = "#FFD92F", Layer3 ="#A6D854", Layer4 = "#66C2A5", Layer5 = "#00A9FF",
                  Layer6="#8DA0CB",WM="#E78AC3")


#######
library(RColorBrewer)
library(ggplot2)
# library(readbitmap)#
myPalette_ <- colorRampPalette(rev(brewer.pal(11, "Spectral")))
# https://ggplot2-book.org/scale-colour.html
myPalette = scale_fill_brewer(palette = "Set2")

#######
# myPalette <- colorRampPalette(rev(brewer.pal(11, "Spectral")))
setwd("/dcs04/hansen/data/ywang/ST/DLPFC/PASTE_out_dist005_szMean/")

# setwd("/dcs04/hansen/data/ywang/ST/DLPFC/PASTE_out_keepAll/")
dir_processedData = "/dcs04/hansen/data/ywang/ST/DLPFC/processed_Data_keepAll/"

# list_sample_pair=list(c(1,2),c(2,3),
                      # c(3,4),c(5,6),c(6,7),c(7,8),c(9,10),c(10,11),c(11,12))
list_sample_pair=list(c(1,2),
                      c(3,4),c(5,6),c(7,8),c(9,10),c(11,12))

#######################################
list_layer=list()
# list_rownames_kpSpots_=list()

  
  ####################################### s1
for(kpair in 1:length(list_sample_pair)){
    list_layer[[kpair]]=list()
    list_rownames_kpSpots_[[kpair]]=list()
    
    sample_s1=list_sample_pair[[kpair]][1]
    sample_s2=list_sample_pair[[kpair]][2]
    # rownames_allSpots = readRDS(paste0(dir_processedData,"rownames_allSpots",
                                       # "_S",sample_s1,"_",sample_s2,"_filterDist005_corrected.rds"))
    
    # rownames_kpSpots_s1= readRDS(paste0(dir_processedData,"rownames_kpSpots",
    #                                     "_S",sample_s1,"_",sample_s2,"_filterDist005_s1_corrected.rds"))
    # rownames_kpSpots_s2= readRDS(paste0(dir_processedData,"rownames_kpSpots",
    #                                     "_S",sample_s1,"_",sample_s2,"_filterDist005_s2_corrected.rds"))
    # 
    # rownames_kpSpots = c(rownames_kpSpots_s1, rownames_kpSpots_s2)
    # length(rownames_kpSpots)
    # # 7112
    
    # load layers
    # save(layer,
         # file=paste0("layer_sample_Sample",j,".RData"))
    load(paste0("//dcs04/hansen/data/ywang/archive/scRNA/Oct5_2021_Lukas_data_more_Genes/out/layer_sample_Sample",sample_s1,".RData"))
    # save(layer_sample_tmp,file=paste0("layer_sample",j,".RData"))
    layer_sample_tmp__s1=layer[]
    
    load(paste0("//dcs04/hansen/data/ywang/archive/scRNA/Oct5_2021_Lukas_data_more_Genes/out/layer_sample_Sample",sample_s2,".RData"))
    # save(layer_sample_tmp,file=paste0("layer_sample",j,".RData"))
    layer_sample_tmp__s2=layer[]
    
    layer_sample_tmp__=c(layer_sample_tmp__s1,layer_sample_tmp__s2)
    # names(layer_sample_tmp__)=rownames_allSpots
    
    # layer_sample_tmp__s1=layer_sample_tmp__[rownames_kpSpots_s1]
    # layer_sample_tmp__s2=layer_sample_tmp__[rownames_kpSpots_s2]
    
    # layer_sample_tmp__ = c(layer_sample_tmp__s1, layer_sample_tmp__s2)
    # length(layer_sample_tmp__)

    list_layer[[kpair]][[1]]=layer_sample_tmp__s1
    list_layer[[kpair]][[2]]=layer_sample_tmp__s2
    

    
    # list_rownames_kpSpots_[[kpair]][[1]]=rownames_kpSpots_s1
    # list_rownames_kpSpots_[[kpair]][[2]]=rownames_kpSpots_s2
    
  }

make_plot<-function(pos,factor_mat,range_perFactor_,
                    layer_sample_tmp___,samplename_){
  plots_l = list()
  for (i in 1:ncol(factor_mat)) {
    # dim(a)
    df_tmp=data.frame( imagerow=pos[,1],
                       imagecol=pos[,2],
                       fill_tmp=factor_mat[,i],
                       layer=layer_sample_tmp___)
    df_tmp=df_tmp[!is.na(df_tmp$layer),]
    df_tmp$layer = factor(df_tmp$layer, levels = c('WM',"Layer6","Layer5","Layer4","Layer3", "Layer2","Layer1"))
    
    
    plot_tmp =  ggplot(df_tmp,aes(x=imagecol,y=imagerow,fill=fill_tmp)) +
      # geom_spatial(data=images_tibble[i,], aes(grob=grob), x=0.5, y=0.5)+
      geom_point(shape = 21, colour = "black", size = 2, stroke = NA)+
      coord_cartesian(expand=FALSE)+
      # scale_fill_gradientn(colours = myPalette_(100))+
      scale_fill_gradientn(colours = myPalette_(100), limits=range_perFactor_[[i]])+
      
      xlab("") +
      ylab("") +
      ggtitle(paste0(samplename_,", mNSF factor ",i))+
      labs(fill = paste0(" "))+
      theme_set(theme_bw(base_size = 10))+
      theme(panel.grid.major = element_blank(),
            panel.grid.minor = element_blank(),
            panel.background = element_blank(),
            axis.line = element_line(colour = "black"),
            axis.text = element_blank(),
            axis.ticks = element_blank())
    
    # if(type_tmp=="l"){
    plots_l[[i]]=plot_tmp
    # }else{
    
    
  }
  plots_l[[i+1]]=ggplot(df_tmp,aes(x=imagecol,y=imagerow,fill=layer)) +
    geom_point(shape = 21, colour = "black", size = 2, stroke = NA)+
    coord_cartesian(expand=FALSE)+
    xlab("") +
    ylab("") +
    ggtitle(paste0("layer "))+
    labs(fill = paste0(" "))+
    theme_set(theme_bw(base_size = 10))+
    theme(panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.background = element_blank(),
          axis.line = element_line(colour = "black"),
          axis.text = element_blank(),
          axis.ticks = element_blank())+
    
    scale_fill_manual(values=group.colors)
  plots_l
}

range_perFactor = list()
for(ksample in 1:12){
  factor_mat_tmp=read.csv(paste0("/dcs04/hansen/data/ywang/ST/DLPFC/PASTE_out_keepAll_scTransform/factors_nb_szMean_sample_s",ksample,"_L10_fullData.csv"),header=T)
  factor_mat_tmp=factor_mat_tmp[,-1]
  for(l in 1:ncol(factor_mat_tmp)){
    if(ksample==1){ range_perFactor[[l]]=range(factor_mat_tmp[,l])}else{
      range_perFactor[[l]]=range(c(range_perFactor[[l]], factor_mat_tmp[,l]))
    }
    
  }
}

pdf(paste0("LdaFalse_L10_szMean_s1To12_fullData.pdf"),height=3,width=18/6*8*3.2)
for(kpair in 1:6){
  for(k in 1:2){
    ksample = (kpair-1)*2+ k 
    a_=read.csv(paste0("/dcs04/hansen/data/ywang/ST/DLPFC/PASTE_out_keepAll_scTransform/factors_nb_szMean_sample_s",ksample,"_L10_fullData.csv"),header=T)
    a_ = a_[,-1]
    layers=list_layer[[kpair]][[k]]
    X=read.csv(paste0('//dcs04/hansen/data/ywang/ST/DLPFC/processed_Data///X_allSpots_sample',ksample,'.csv'))
    p1= make_plot(X, a_, range_perFactor,layers, paste0("sample ",ksample))
    print(plot_grid(plotlist = p1,nrow=1))
  }
}
dev.off()




pdf(paste0("LdaFalse_L10_szMean_s1To12_s1_5_7_9_fullData.pdf"),height=3,width=18/6*8*3.2)
for(kpair in c(1,3,4,5)){
  for(k in 1){
    ksample = (kpair-1)*2+ k 
    a_=read.csv(paste0("/dcs04/hansen/data/ywang/ST/DLPFC/PASTE_out_keepAll_scTransform/factors_nb_szMean_sample_s",ksample,"_L10_fullData.csv"),header=T)
    a_ = a_[,-1]
    layers=list_layer[[kpair]][[k]]
    rownames_kpSpots=list_rownames_kpSpots_[[kpair]][[k]]
    p1= make_plot(rownames_kpSpots, a_,range_perFactor, layers, paste0("sample ",ksample))
    print(plot_grid(plotlist = p1,nrow=1))
  }
}
dev.off()







