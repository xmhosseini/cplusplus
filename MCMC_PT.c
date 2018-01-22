#include <iostream>
#include <Eigen/Dense>
#include <stdlib.h>     //srand, rand
#include <time.h>       // time
#include "math.h"       // for RAND, and rand, log and sqrt
#include <ctime>
using namespace std;    // for cout
double sampleNormal();  // function declaration
using namespace Eigen;

double mypdf     (double x1,double x2,double x3);
double expectancy(double x1,double x2,double x3);

const double pi = 3.1415926535897;
const double TUNED_VAR = 0.199;

int main()
{


float exe_time=0;
float t1=clock();
double TOSS_A_DICE;

double accepted=0;
double rejected=0;
double accepted2=0;
double rejected2=0;
double exchanged=0;

double EX_MCMC = 0;
double y1_c=1;
double y2_c=1;
double y3_c=1;

double y1_c2=1;
double y2_c2=1;
double y3_c2=1;

// initialization
double y1_now=1;
double y2_now=1;
double y3_now=1;

double y1_now2=1;
double y2_now2=1;
double y3_now2=1;

double exchange1=0;
double exchange2=0;
double exchange3=0;

double i=1;
double criteria=0;


//initialize random seed
srand (time(NULL));


while (criteria<10000)
{
// chain 1 and the main 1
y1_c = y1_now + TUNED_VAR*sampleNormal();
y2_c = y2_now + TUNED_VAR*sampleNormal();
y3_c = y3_now + TUNED_VAR*sampleNormal();

TOSS_A_DICE = (rand() % 10000) / 10000.0; // random between 0 and 1
if (TOSS_A_DICE < mypdf(y1_c,y2_c,y3_c) /mypdf(y1_now,y2_now,y3_now) )
{
	y1_now = y1_c;
    y2_now = y2_c;
    y3_now = y3_c;

    accepted=accepted+1;

}
else
{
    rejected=rejected+1;
}



//chain 2 and use of pow(base,exponent)
y1_c2 = y1_now2 + TUNED_VAR*sampleNormal();
y2_c2 = y2_now2 + TUNED_VAR*sampleNormal();
y3_c2 = y3_now2 + TUNED_VAR*sampleNormal();

TOSS_A_DICE = (rand() % 10000) / 10000.0; // random between 0 and 1
if (TOSS_A_DICE < pow(mypdf(y1_c2,y2_c2,y3_c2),0.25) /pow(mypdf(y1_now2,y2_now2,y3_now2),0.25) )
{
	y1_now2 = y1_c2;
    y2_now2 = y2_c2;
    y3_now2 = y3_c2;

    accepted2=accepted2+1;

}
else
{
    rejected2=rejected2+1;
}

// exchange
TOSS_A_DICE = (rand() % 10000) / 10000.0; // random between 0 and 1
   if (TOSS_A_DICE < ((mypdf(y1_now2,y2_now2,y3_now2)*(pow(mypdf(y1_now,y2_now,y3_now),0.25))) / (mypdf(y1_now,y2_now,y3_now)*( pow(mypdf(y1_now2,y2_now2,y3_now2),0.25)))))
       {
	   exchange1 = y1_now2;
	   exchange2 = y2_now2;
	   exchange3 = y3_now2;

       y1_now2  = y1_now;
       y2_now2  = y2_now;
       y3_now2  = y3_now;

       y1_now = exchange1;
       y2_now = exchange2;
       y3_now = exchange3;

	   exchanged = exchanged + 1;
	   }



EX_MCMC=EX_MCMC+expectancy(y1_now,y2_now,y3_now);

if (((EX_MCMC/i)<8.05) && ((EX_MCMC/i)>7.95)) {criteria=criteria+1;} else {criteria=0;}

i=i+1;

cout << "Accepted Chain1: " <<   accepted << endl;
cout << "Rejected Chain1: " <<   rejected << endl;
cout << "Accepted Chain2: " <<   accepted2 << endl;
cout << "Rejected Chain2: " <<   rejected2 << endl;
cout << "Exchanged b/w 2 Chains: " <<   exchanged << endl;
cout<<EX_MCMC/i<<endl;

}
float t2 = clock();
exe_time = (t2-t1)/double(CLOCKS_PER_SEC);


cout << "Converged Value: " <<  EX_MCMC/i << endl;
cout<<"Execution time: "    <<  exe_time << endl;


return 0;

}


double sampleNormal() {
    double u = ((double) rand() / (RAND_MAX)) * 2 - 1;
    double v = ((double) rand() / (RAND_MAX)) * 2 - 1;
    double r = u * u + v * v;
    if (r == 0 || r > 1) return sampleNormal();
    double c = sqrt(-2 * log(r) / r);
    return u * c;
}

double mypdf (double x1, double x2, double x3)
{
double pdf;

//double f1 = (exp(-(pow(((x1-1) /1),2))/2))/1/sqrt(2*pi);
//double f2 = (exp(-(pow(((x2-1) /1),2))/2))/1/sqrt(2*pi);
//double f3 = (exp(-(pow(((x3-1) /1),2))/2))/1/sqrt(2*pi);

MatrixXd sig(3,3);
sig << 1, 0.5, 0,
       0.5, 1, 0,
       0, 0, 1;

MatrixXd siginverse(3,3);
siginverse=sig.inverse();


MatrixXd mu(3,1);
mu(0,0) = 1;
mu(1,0) = 1;
mu(2,0) = 1;

MatrixXd x(3,1);
x(0,0) = x1;
x(1,0) = x2;
x(2,0) = x3;

double sigdet = sig.determinant();

MatrixXd xminmu(3,1);
xminmu = x - mu;
MatrixXd xminmutranspose(1,3);
xminmutranspose=xminmu.transpose();

MatrixXd xminmutsiginv(1,3);
xminmutsiginv=xminmutranspose*siginverse;
MatrixXd epowered(1,1);
epowered = xminmutsiginv*xminmu;

double epower = (-0.5)*epowered(0,0);


pdf = exp(epower)/sqrt(2*pi*2*pi*2*pi*sigdet);

return pdf;
}

double expectancy (double x1, double x2, double x3)
{
double expected;

expected = x1*x2;

return expected;
}


