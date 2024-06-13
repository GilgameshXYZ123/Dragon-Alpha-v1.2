/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.websocket.message;

/**
 *
 * @author dell
 */
public class P2PMsg extends Message
{
    //columns-------------------------------------------------------------------
    String dstPartitionId;
    String dstSessionId;
    
    //functions-----------------------------------------------------------------
    public P2PMsg() 
    {
        super(HEAD_P2P);
    }
    public P2PMsg(String destPartitionId, String dstSessionId,
            String srcPartitionId, String srcSessionId, String data) 
    {
        super(HEAD_P2P, srcPartitionId, srcSessionId, data);
        this.dstPartitionId = destPartitionId;
        this.dstSessionId = dstSessionId;
    }
    public String getDstPartitionId() 
    {
        return dstPartitionId;
    }
    public void setDstPartitionId(String dstPartitionId) 
    {
        this.dstPartitionId = dstPartitionId;
    }
    public String getDstSessionId() 
    {
        return dstSessionId;
    }
    public void setDstSessionId(String dstSessionId) 
    {
        this.dstSessionId = dstSessionId;
    }
    @Override
    public String toString() 
    {
        return super.toString()+':'+dstPartitionId+":"+dstSessionId;
    }
    @Override
    public void loadData(String[] args) 
    {
        super.loadData(args);
        dstPartitionId=args[4];
        dstSessionId=args[5];
    }
}
