"""Independent transformed-hypergeometric checks; reads only."""
from pathlib import Path
import hashlib,json
import mpmath as mp
mp.mp.dps=80
root=Path(__file__).resolve().parent
data=json.loads((root/'expected.json').read_text());checks=0

def m(v):
 n,d=float(v).as_integer_ratio();return mp.mpf(n)/d

def expect(actual,expected):
 global checks
 assert float(actual)==expected,(mp.nstr(actual,25),expected)
 checks+=1

for row in data['cases']:
 p=row['parameters'];h=row['history'];r,alpha,a,b=map(m,[p[k] for k in ['r','alpha','a','b']]);x=h['x'];B=alpha+m(h['t']);q=a-1
 alive=mp.mpf(1) if not x else 1/(1+a/(b+x-1)*((alpha+m(h['t']))/(alpha+m(h['tx'])))**(r+x))
 def conditional(H):
  if not H:return mp.mpf(0)
  z=H/(B+H)
  def numerator(delta):
   c=delta+b+x
   return 1-mp.exp(delta*mp.log1p(-z))*mp.hyp2f1(delta+b-r,delta,c,z)
  return (q+b+x)*(mp.diff(numerator,0) if q==0 else numerator(q)/q)
 for pred in row['predictions']:
  H=m(pred['horizon']);v=conditional(H)
  expect(alive,pred['probability_active']);expect(v,pred['expected_if_active']);expect(alive*v,pred['expected_purchases'])

if (root/'manifest.json').exists():
 manifest=json.loads((root/'manifest.json').read_text())
 for name,rec in manifest['files'].items():
  raw=(root/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==rec['sha256'] and len(raw)==rec['bytes'];checks+=1
print(json.dumps({'status':'passed','checks':checks,'cases':len(data['cases']),'precision_digits':80,'method':'Independent transformed hypergeometric identity and analytic numerator derivative','native_calls':0}))
