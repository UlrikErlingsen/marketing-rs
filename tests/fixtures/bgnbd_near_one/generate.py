import json,math,struct,sys
from fractions import Fraction
import mpmath as mp
from pathlib import Path
DIGITS=int(sys.argv[1]);mp.mp.dps=DIGITS

def exact(v):
 n,d=float(v).as_integer_ratio();return mp.mpf(n)/d
cases=[]
for q in ['-0.0001','0.0001','-0.000001','0.000001','-0.00000001','0.00000001','-0.0000000001','0.0000000001','-0.000000000001','0.000000000001','0']:
 a=float(Fraction(1)+Fraction(q))
 for x,tx in [(0,0.),(3,8.25)]:
  r,alpha,b,T=1.1,5.25,2.8,19.5
  p=[exact(v) for v in [r,alpha,a,b]];rr,aa,ap,bp=p;k=rr+x;B=aa+exact(T)
  alive=mp.mpf(1) if not x else 1/(1+ap/(bp+x-1)*((aa+exact(T))/(aa+exact(tx)))**k)
  values=[]
  horizons=[0.,1e-8,26.,1000.]
  for h in horizons:
   H=exact(h)
   if not h:value=mp.mpf(0)
   else:
    def f(u):
     if u==0:return mp.mpf(0)
     return -mp.expm1(-k*mp.log1p(u*H/B))*u**(ap-2)*(1-u)**(bp+x-1)/mp.beta(ap,bp+x)
    value=mp.quad(f,[0,mp.mpf('.1'),mp.mpf('.5'),1])
   values.append({'horizon':h,'probability_active':float(alive),'expected_if_active':float(value),'expected_purchases':float(alive*value)})
  cases.append({'id':f'q_{q}_x_{x}','decimal_offset_label':q,'parameters':{'r':r,'alpha':alpha,'a':a,'b':b},'a_hex':a.hex(),'history':{'x':x,'tx':tx,'t':T,'weight':1.},'predictions':values})
out={'schema':'bgnbd-near-one-synthetic-posterior-integral-v1','input_semantics':'Each decimal JSON numeric input denotes its nearest binary64, integrated at its exact rational value. Offset labels are descriptive; a_hex pins actual a.','primary_source':'https://doi.org/10.1287/mksc.1040.0098','derivation':'Conditional active lambda posterior is Gamma(r+x,alpha+T), dropout posterior Beta(a,b+x). E future|active = integral (1-(1+p*h/(alpha+T))^-(r+x))/p times the beta posterior. Evaluate numerator with expm1/log1p. Multiply by posterior active probability.','method':'mpmath tanh-sinh posterior integral; no Rust or native evaluation','tolerances':{'scalar':{'absolute':1e-12,'relative':1e-10}},'cases':cases}
print(json.dumps(out,indent=2,allow_nan=False))
