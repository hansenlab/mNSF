


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
# library(readbitmap)#
# devtools::install_github('exaexa/scattermore')
library(scattermore)

myPalette_ <- colorRampPalette(rev(brewer.pal(11, "Spectral")))
# https://ggplot2-book.org/scale-colour.html
myPalette = scale_fill_brewer(palette = "Set2")

dir_out="//dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal_spaceRanger1_1_0/out//"
# dir.create(dir_out)
setwd(dir_out)

##### this data (two-sample data) is used as seurat examples:
# https://satijalab.org/seurat/articles/spatial_vignette.html

##### ST plot
plots_l=list()
for (i in c(1:20)) {
  # _500selectedFeatures_dev_interpolated_35percent_szMean.csv
  for (sample_ in 1:4){#NPFH_NPF_sample2_v2_
    # /dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal_spaceRanger1_1_0/out/
    a=read.csv(paste0("factors_sample",sample_,"_500selectedFeatures_dev_interpolated_35percent_szMean.csv"),
               header=T)
    a=a[,-1]
    # "factors_sample"+str(k+1)+"_500selectedFeatures_dev_interpolated_35percent.csv"
    # NPF_sample1_L12_v2_fulldata_twoSample___addmarkergenes
    dim(a)
    X_allSpots=read.csv(paste0("X_sample",sample_,".csv"))
    
    
    X=X_allSpots
    dim(X_allSpots)

    
    dim(a)
    sample_label=sample_
    if(sample_==2){sample_label=3}
    if(sample_==3){sample_label=2}
    
    df_tmp=data.frame(imagecol=X[,2],
                      imagerow=-X[,1],
                      fill_tmp=a[,i],
                      sample=sample_label)
    if(sample_==1){df_=df_tmp}else{
      df_=rbind(df_,df_tmp)
    }
    
    
  }
  
  df_$sample[df_$sample==1] = "Anterior, S1"
  df_$sample[df_$sample==2] = "Posterior, S1"
  df_$sample[df_$sample==3] = "Anterior, S2"
  df_$sample[df_$sample==4] = "Posterior, S2"
  
  df_$sample = factor(df_$sample, levels= c("Anterior, S1","Posterior, S1","Anterior, S2","Posterior, S2"))
  plot_tmp =  ggplot(df_,aes(x=imagecol,y=imagerow,fill=fill_tmp)) +
  # plot_tmp =  ggplot(df_) +
    # geom_spatial(data=images_tibble[i,], aes(grob=grob), x=0.5, y=0.5)+
    geom_point(shape = 21, colour = "black", size = .75, stroke = NA)+
    # geom_scattermore(shape = 21, colour = "black", size = 1, stroke = 0.5)+
    # geom_scattermore(aes(x=imagecol,y=imagerow,col=fill_tmp),
                     # size=3,
      # pointsize = 10
                       # pixels = c(1000, 1000),
                       # interpolate = F
    # ) +
    coord_cartesian(expand=FALSE)+
    # scale_colour_brewer(palette = "Set1")+
    # scale_color_viridis_c()+
    scale_fill_gradientn(colours = myPalette_(100))+
    xlab("") +
    ylab("") +
    ggtitle(paste0("M",i))+
    labs(fill = paste0(" "))+
    theme_set(theme_bw(base_size = 10))+
    theme(panel.background = element_rect(fill = 'black', color = 'black'),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          # panel.background = element_blank(),
          axis.line = element_line(colour = "black"),
          axis.text = element_blank(),
          axis.ticks = element_blank())+
    facet_wrap(~sample,ncol=2)#+

  plots_l[[i]]=plot_tmp
  
  
  
  
}

p__=(plot_grid(plotlist = plots_l,nrow=4))

ggsave('4samples_LdaFalse_L20_4samplemnsf_L20_35percentInducedPoints_500HVGsDev_blackBackground_szMean.png', units='in', 
       height=18/7*7.8,width=25,
       p__)




ggsave('4samples_LdaFalse_L20_4samplemnsf_L20_35percentInducedPoints_500HVGsDev_blackBackground_szMean.pdf', 
       height=18/7*7.8*.9,width=25*.9,
       p__)



