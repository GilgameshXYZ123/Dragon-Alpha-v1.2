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
public final class LocalMsg extends Message
{
    //columns-------------------------------------------------------------------
    String dstPartitionId;
    
    //constructor---------------------------------------------------------------
    public LocalMsg()
    {
        super(HEAD_LOCAL);
    }
    public LocalMsg(String dstPartitionId, String srcPartitionId, String srcSessionId, String data) 
    {
        super(HEAD_LOCAL, srcPartitionId, srcSessionId, data);
        this.dstPartitionId = dstPartitionId;
    }
    public String getDstPartitionId() 
    {
        return dstPartitionId;
    }
    public void setDstPartitionId(String dstPartitionId) 
    {
        this.dstPartitionId = dstPartitionId;
    }
    @Override
    public String toString()
    {
        return super.toString()+':'+dstPartitionId;
    }
    @Override
    public void loadData(String[] args) 
    {
        super.loadData(args);
        dstPartitionId=args[4];
    }
}
