def to_html(df, path):
    f = open(path,'w')
    f.write(df.render())
    f.close()
import os
import pandas as pd
import subprocess
df = pd.DataFrame({'a':[1,2,3]}).style
to_html(df,'table.html')
subprocess.call(
    'wkhtmltoimage -f png --width 0 table.html table.png', shell=True)