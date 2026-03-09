from unittest import result
from langchaindemo  import run_simple_chain,run_sequence_chain,run_agent_task
if __name__=="__main__":
    #resultA=run_simple_chain("美食店铺")
    #print(resultA)

    #resultB=run_sequence_chain("量子计算")
    #print(resultB)

    resultC=run_agent_task("计算半径为5的圆的面积")
    print(resultC)
   