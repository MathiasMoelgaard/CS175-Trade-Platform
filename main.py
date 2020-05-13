
from src.agent.simple_agent import simple_agent
from src.agent.agent_thread import agent_thread
from src.trade_platform.trade_platform import trade_platform

if __name__ == "__main__":
    t = trade_platform(length=5000, data_path='sample_data/a.csv', enable_plot=False,random=False)
    t.add_agent(simple_agent())
    t.start()

'''/*
 * Synchronous(default) / Asynchronous(deprecated)
 * Market / Agent 
 *    
 * Trade platform
 *
*/'''
