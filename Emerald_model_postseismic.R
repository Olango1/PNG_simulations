library(mgcv)
#library(ordPens)
#library(Metrics)

#####################################################
#########read data and get target variable###########
#####################################################s
data.raw = sf::st_read("/Users/olangoea/Library/CloudStorage/OneDrive-KAUST/Documents/PNG/please_work/SU_poly1.gpkg")
data.raw[is.na(data.raw)] = 0
data.raw = na.omit(data.raw)
data.raw$lithology = factor(data.raw$lithology_majority)
data.raw <- subset(data.raw,lithology != 0)
data.raw$lithology[data.raw$lithology == 14] = 1

Formula.eva = Postseismic_AmplifR ~   
  s(slope_avg, k = 3)+
  s(relief, k = 3)+
  s(dist_geo, k = 3)+
  s(Compo_pgv, k = 3)+
  s(lithology,bs='re')

Fit.susc.par = mgcv::gam(Formula.eva, family = binomial, method="REML", data = data.raw) 

#save(Fit.susc.par,file=paste0("C:/Users/TanyasH1/OneDrive - Universiteit Twente/Documents/Emerald/R/ModelFit_coseismic.RData"))
#load("C:/Users/TanyasH1/OneDrive - Universiteit Twente/Documents/Emerald/R/ModelFit_coseismic.RData")
plot(Fit.susc.par, trans=plogis, pages=1, all.terms=T, shade=T, ylab="Probability")
#dev.of

## goodness of fit
library(pROC)
dataLabel = data.raw$Coseismic_Amplif
predictedSus = Fit.susc.par$fitted.values
ROC.fit = roc(dataLabel~predictedSus)
ROC.fit$auc
plot(ROC.fit)
#dev.off()

######################
#install.packages("plotrix")
library(plotrix)

# Assuming you have already loaded and fitted the model, and extracted the coefficients and standard errors
coefficients = coef(Fit.susc.par)
cov_matrix = vcov(Fit.susc.par)

# Calculate the 95% confidence intervals
ci_mult <- 1.96
std_errors <- sqrt(diag(cov_matrix))
ci_lower <- coefficients - ci_mult * std_errors # the formula for confidence intervals
ci_upper <- coefficients + ci_mult * std_errors


# Extract the coefficients and confidence intervals for lithology
lithology_effects <- grep("lithology", names(coefficients))
ci_lower_lithology <- ci_lower[lithology_effects]
ci_upper_lithology <- ci_upper[lithology_effects]
lith_coeff <- coefficients[lithology_effects]
lith_std_errors <- std_errors[lithology_effects]

LowerProb.Litho =  exp(ci_lower_lithology)/(1+exp(ci_lower_lithology))
UpperProb.Litho =  exp(ci_upper_lithology)/(1+exp(ci_upper_lithology))
mean_lith_coeff = exp(lith_coeff)/(1+exp(lith_coeff))


unique_values <- c(1,4,5,9,11)
plot(unique_values,mean_lith_coeff, pch = 15, ylim=c(0,1))
points(unique_values,UpperProb.Litho, pch = 20)
points(unique_values,LowerProb.Litho, pch = 20)
#label_lit= c("Uncolsolidated sediment","Mixed sedimentray rock","Carbonate sedimentray rock","Basic volcanics","Intermediate plutonics")

# Define unique values and labels
# Define unique values and labels
unique_values <- c(1, 4, 5, 9, 11)
label_lit <- c("Unconsolidated sediment", "Mixed sedimentary rock", "Carbonate sedimentary rock", 
               "Basic volcanics", "Intermediate plutonics")

# Combine unique values with labels for the legend
legend_labels <- paste(unique_values, label_lit, sep = " - ")

# Create the plot
par(mar=c(4.5, 4.5, 2, 2.3))
plotCI(x=unique_values, y=mean_lith_coeff,
       li=LowerProb.Litho, ui=UpperProb.Litho,
       ylim=c(0.0, 1.0), xlab = "Lithology", ylab = "Partial effects",
       family = "serif", cex.axis = 1.2, cex.lab=1.4, xaxt="n", pch=19, col="red", scol="black",
       lwd=2)

# Add custom x-axis labels
axis(side = 1, at=unique_values, labels=unique_values, cex.axis = 1.2, family = "serif")

# Add a horizontal line at y=0
abline(h=0, col = "gray40", lwd = 2, lty=2)

# Add a legend with combined unique values and labels
legend("topleft", legend=legend_labels, pch=19, col="red", pt.cex=0.4, cex=0.55, bty="n", text.font=2)



