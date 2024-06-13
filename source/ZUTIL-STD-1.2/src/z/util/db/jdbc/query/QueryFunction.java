/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.db.jdbc.query;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import java.util.HashSet;
import java.util.Set;
import z.util.factory.Meta;
import z.util.math.vector.Vector;
import z.util.lang.annotation.Passed;

/**
 *
 * @author dell
 */
public final class QueryFunction 
{
    //constants-----------------------------------------------------------------
    private static final String F_AVG="avg";
    private static final String F_STDDEV="stddev";
    private static final String F_MAX="max";
    private static final String F_MIN="min";
    
    public static final String JDBC_NORMALIZE_AVG="jdbc.normalize.avg";
    public static final String JDBC_NORMALIZE_STDDEV="jdbc.normalize.avg";
    
    private static final String JDBC_TOPSIS_AVG_FIRST="jdbc.topsis.avg.first";
    private static final String JDBC_TOPSIS_STDDEV_FIRST="jdbc.topsis.stddev.first";
    
    private static final String JDBC_TOPSIS_MAX="jdbc.topsis.max";
    private static final String JDBC_TOPSIS_MIN="jdbc.topsos.min";
    
    private static final String JDBC_TOPSIS_AVG_SECOND="jdbc.topsis.avg.second";
    private static final String JDBC_TOPSIS_STDDEV_SECOND="jdbc.topsis.stddev.second";
    
    //functions-----------------------------------------------------------------
    private QueryFunction() {}
    //<editor-fold defaultstate="collapsed" desc="Desc-Function">
    //<editor-fold defaultstate="collapsed" desc="class ColumnInfo TableInfo">
    public static final class ColumnInfo
    {
        //columns---------------------------------------------------------------
        String tableName;
        String name;
        String label;
        String typeName;
        int scale;
        int precision;
        
        //functions-------------------------------------------------------------
        ColumnInfo(){}
        public String getTableName() 
        {
            return tableName;
        }
        public String getName() 
        {
            return name;
        }
        public String getLabel() 
        {
            return label;
        }
        public String getTypeName() 
        {
            return typeName;
        }
        public int getScale() 
        {
            return scale;
        }
        public int getPrecision() 
        {
            return precision;
        }
        public void toString(StringBuilder sb)
        {
            sb.append(tableName).append('.').append(label).append('(').append(name).append(" )\t")
              .append(typeName).append('(').append(scale).append(", ").append(precision).append(')');
        }
        @Override
        public String toString()
        {
            StringBuilder sb=new StringBuilder();
            this.toString(sb);
            return sb.toString();
        }
    }
    public static final class TableInfo
    {
        //columns---------------------------------------------------------------
        HashSet<String> tables=new HashSet<>();
        ColumnInfo[] columns;
        
        //functions-------------------------------------------------------------
        TableInfo(ResultSetMetaData rsmt) throws SQLException   
        {
            int len=rsmt.getColumnCount();
            columns=new ColumnInfo[len];
            for(int i=0;i<len;i++)
            {
                columns[i]=new ColumnInfo();
                columns[i].tableName=rsmt.getTableName(i+1);
                columns[i].name=rsmt.getColumnName(i+1);
                columns[i].label=rsmt.getColumnLabel(i+1);
                columns[i].typeName=rsmt.getColumnTypeName(i+1);
                columns[i].scale=rsmt.getScale(i+1);
                columns[i].precision=rsmt.getPrecision(i+1);
                
                tables.add(columns[i].tableName);
            }
        }
        public Set<String> getTables() 
        {
            return tables;
        }
        public ColumnInfo[] getColumns() 
        {
            return columns;
        }
        @Override
        public String toString()
        {
            StringBuilder sb=new StringBuilder();
            sb.append("tables: ").append(tables).append('\n');
            sb.append("[column]\t[type]\n");
            for(int i=0;i<columns.length;i++) 
            {
                columns[i].toString(sb);
                sb.append('\n');
            }
            return sb.toString();
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Inner-Code">
    @Passed
    private static TableInfo innerDesc(PreparedStatement pst, String table) throws SQLException 
    {
        TableInfo result=null;
        ResultSet rs=null;
        try
        {
            rs=pst.executeQuery("select * from  "+table+" limit 0");
            result=new TableInfo(rs.getMetaData());
        }
        finally
        {
           if(rs!=null) rs.close();
        }
        return result;
    }
    @Passed
    private static Set<String> innerGetTableColNames(PreparedStatement pst, String table) throws SQLException
    {
        HashSet<String> cols=new HashSet<>();
        ResultSet rs=null;
        try
        {
            rs=pst.executeQuery("select * from "+table+" limit 0");
            ResultSetMetaData rsmt=rs.getMetaData();
            for(int i=1,len=rsmt.getColumnCount();i<=len;i++)
                cols.add(rsmt.getColumnName(i));
        }
        finally
        {
            if(rs!=null) rs.close();
        }
        return cols;
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="For Table Operation">
    //<editor-fold desc="Inner-Code">
    @Passed
    private static boolean innnerCopyTable(PreparedStatement pst, String srcTable, String dstTable) 
            throws SQLException 
    {
        StringBuilder sb=new StringBuilder();
        sb.append("create table if not exists ").append(dstTable).append(" as select * from ").append(srcTable);
        return pst.execute(sb.toString());
    }
    @Passed
    private static boolean innerDropTable(PreparedStatement pst, String table) throws SQLException
    {
        return pst.execute("drop table "+table);
    }
    @Passed
    private static boolean innerTruncate(PreparedStatement pst, String table) throws SQLException
    {
        return pst.execute("truncate "+table);
    }
    @Passed
    private static boolean innerAlterTable(PreparedStatement pst, String table, String query) throws SQLException
    {
        StringBuilder sb=new StringBuilder();
        sb.append("alter table ").append(table).append('\n').append(query);
        return pst.execute(sb.toString());
    }
    @Passed
    private static boolean innnerAlterTable(PreparedStatement pst, String table, String[] query) throws SQLException
    {
        StringBuilder sb=new StringBuilder();
        sb.append("alter table ").append(table).append('\n');
        Vector.append(sb, "\n", query);
        return pst.execute(sb.toString());
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Aggregation-Function">
    //<editor-fold defaultstate="collapsed" desc="Inner-Code">
    @Passed
    private static void innerExecuteAggregation(PreparedStatement pst, String table, String function, String[] columns, double[] result) 
            throws SQLException
    {
        ResultSet rs=null;
        try
        {
            String funcHead=function+'(';
            String colHead=function+'_';
            String funcTail=") as "+colHead;
            //create query------------------------------------------------------
            StringBuilder sb=new StringBuilder();
            sb.append("select ")
              .append(funcHead).append(columns[0]).append(funcTail).append(columns[0]);
            
            for(int i=1;i<columns.length;i++)
                sb.append(',').append(funcHead).append(columns[i]).append(funcTail).append(columns[i]);
            
            sb.append(" from ").append(table);
            //execute-----------------------------------------------------------
            rs=pst.executeQuery(sb.toString());
            if(rs.next())
            for(int i=0;i<columns.length;i++)
                result[i]=rs.getDouble(colHead+columns[i]);
        }
        finally
        {
            if(rs!=null) rs.close();
        }
    }
    @Passed
    private static boolean innerNormalize(PreparedStatement pst,String table, String[] columns, double[] avg, double[] stddev) 
            throws SQLException
    {
        //get avg and stddev----------------------------------------------------
        QueryFunction.innerExecuteAggregation(pst, table, F_AVG, columns, avg);
        QueryFunction.innerExecuteAggregation(pst, table, F_STDDEV, columns, stddev);
            
        //do normalize----------------------------------------------------------
        StringBuilder sb=new StringBuilder();
        sb.append("update ").append(table).append(" set ");
            
        sb.append(columns[0]).append("=(").append(columns[0]).append('-').append(avg[0]).append(")/").append(stddev[0]);
            
        for(int i=1;i<columns.length;i++)
        sb.append(',')
          .append(columns[i]).append("=(").append(columns[i]).append('-').append(avg[i]).append(")/").append(stddev[i]);
            
        return pst.execute(sb.toString());
    }
    @Passed
    private static boolean innerNormalize(PreparedStatement pst, String srcTable, String dstTable, String[] columns, double[] avg, double[] stddev) 
            throws SQLException
    {
        //copy table to protect the original table------------------------------
        QueryFunction.innnerCopyTable(pst, srcTable, dstTable);
        
        //get avg and stddev----------------------------------------------------
        QueryFunction.innerExecuteAggregation(pst, dstTable, F_AVG, columns, avg);
        QueryFunction.innerExecuteAggregation(pst, dstTable, F_STDDEV, columns, stddev);
            
        StringBuilder sb=new StringBuilder();
        sb.append("update ").append(dstTable).append(" set ");
            
        sb.append(columns[0]).append(" = (").append(columns[0]).append('-').append(avg[0]).append(")/").append(stddev[0]);
            
        for(int i=1;i<columns.length;i++)
        sb.append(',').append(columns[i]).append(" = (").append(columns[i]).append('-').append(avg[i]).append(")/").append(stddev[i]);
            
        //execute query-----------------------------------------------------
        return pst.execute(sb.toString());
    }
    @Passed
    private static void innerTopsis(PreparedStatement pst, String srcTable, String dstTable, String[] columns, Meta mt) 
            throws SQLException
    {
        double[] avg1=new double[columns.length];
        double[] stddev1=new double[columns.length];
        
        //first normalize-------------------------------------------------------
        QueryFunction.innerNormalize(pst, srcTable, dstTable, columns, avg1, stddev1);
        mt.put(JDBC_TOPSIS_AVG_FIRST, avg1);
        mt.put(JDBC_TOPSIS_STDDEV_FIRST, stddev1);
        
        //find ma and min-------------------------------------------------------
        double[] max=new double[columns.length];
        double[] min=new double[columns.length];
        QueryFunction.innerExecuteAggregation(pst, dstTable, F_MAX, columns, max);
        QueryFunction.innerExecuteAggregation(pst, dstTable, F_MIN, columns, min);
        mt.put(JDBC_TOPSIS_MAX, max);
        mt.put(JDBC_TOPSIS_MIN, min);
        
        //add topsis columns----------------------------------------------------
        Set<String> cols=QueryFunction.innerGetTableColNames(pst, dstTable);
        String[] topsisCols={"d_min","d_max","topsis_s"};
        for(int i=0;i<topsisCols.length;i++)
            if(!cols.contains(topsisCols[i]))
                QueryFunction.innerAlterTable(pst, dstTable, "add column "+topsisCols[i]+" double");//kindly executed in batch
        
        
        String entry=null;
        StringBuilder sb=new StringBuilder();
        //compute min_dis and max_dis-------------------------------------------
        sb.append("update ").append(dstTable).append(" set\n\td_min=sqrt(");
        entry='('+columns[0]+(min[0]>0?  '-':'+')+min[0]+')';
        sb.append(entry).append('*').append(entry);
        for(int i=1;i<columns.length;i++)
        {
            entry='('+columns[i]+(min[i]>0?  '-':'+')+min[i]+')';
            sb.append('+').append(entry).append('*').append(entry);
        }
        
        sb.append("),\n\td_max=sqrt(");
        entry='('+columns[0]+(max[0]>0?  '-':'+')+max[0]+')';
        sb.append(entry).append('*').append(entry);
        for(int i=1;i<columns.length;i++)
        {
            entry='('+columns[i]+(max[i]>0?  '-':'+')+max[i]+')';
            sb.append('+').append(entry).append('*').append(entry);
        }
        sb.append(')');  
        pst.execute(sb.toString());
            
        //compute topsis_s------------------------------------------------------
        sb=new StringBuilder();
        sb.append("update ").append(dstTable).append(" set topsis_s=d_min/(d_min+d_max)");
        pst.execute(sb.toString());
        
        //second normalize------------------------------------------------------
        double[] avg2=new double[topsisCols.length];
        double[] stddev2=new double[topsisCols.length];
        
        QueryFunction.innerNormalize(pst, dstTable, topsisCols, avg2, stddev2);
        mt.put(JDBC_TOPSIS_AVG_SECOND, avg2);
        mt.put(JDBC_TOPSIS_STDDEV_SECOND, stddev2);
    }
    //</editor-fold>
    public static double[] aggregate(Connection con, String table, String function, String[] columns)
    {
        PreparedStatement pst=null;
        double[] result=null;
        try
        {
            pst=con.prepareStatement("");
            result=new double[columns.length];
            QueryFunction.innerExecuteAggregation(pst, table, function, columns, result);
        }
        catch(SQLException e)
        {
            e.printStackTrace();
            throw new RuntimeException(e);
        }
        finally
        {
            try
            {
                if(pst!=null) pst.close();
            }
            catch(Exception e) {throw new RuntimeException(e);}
        }
        return result;
    }
    public static Meta aggregate(Connection con, String table, String[] function, String[] columns)
    {
        PreparedStatement pst=null;
        Meta mt=null;
        try
        {
            pst=con.prepareStatement("");
            double[] result=null;
            mt=new Meta();
            for(int i=0;i<function.length;i++)
            {
                result=new double[columns.length];
                QueryFunction.innerExecuteAggregation(pst, table, function[i], columns, result);
                mt.put(function[i], result);
            }
        }
        catch(SQLException e)
        {
            e.printStackTrace();
            throw new RuntimeException(e);
        }
        finally
        {
            try
            {
                if(pst!=null) pst.close();
            }
            catch(SQLException e) {throw new RuntimeException(e);}
        }
        return mt;
    }
    public static double[] average(Connection con, String table, String[] columns)
    {
        return QueryFunction.aggregate(con, table, F_AVG, columns);
    }
    public static double[] stddev(Connection con, String table, String[] columns)
    {
        return QueryFunction.aggregate(con, table, F_STDDEV, columns);
    }
    public static double[] max(Connection con, String table, String[] columns)
    {
        return QueryFunction.aggregate(con, table, F_MAX, columns);
    }
    public static double[] min(Connection con, String table, String[] columns)
    {
        return QueryFunction.aggregate(con, table, F_MIN, columns);
    }
    public static Meta normalize(Connection con, String table, String[] columns)
    {
        PreparedStatement pst=null;
        Meta mt=null;
        try
        {
            pst=con.prepareStatement("");
            mt=new Meta();
            double[] avg=new double[columns.length];
            double[] stddev=new double[columns.length];
            QueryFunction.innerNormalize(pst, table, columns, avg, stddev);
            mt.put(JDBC_NORMALIZE_AVG, avg);
            mt.put(JDBC_NORMALIZE_STDDEV, stddev);
        }
        catch(SQLException e)
        {
            e.printStackTrace();
            throw new RuntimeException(e);
        }
        finally
        {
            try
            {
                if(pst!=null) pst.close();
            }
            catch(SQLException e) {throw new RuntimeException(e);}
        }
        return mt;
    }
    public static Meta topsis(Connection con, String srcTable, String dstTable, String[] columns)
    {
        PreparedStatement pst=null;
        Meta mt=null;
        try
        {
            pst=con.prepareStatement("");
            mt=new Meta();
            QueryFunction.innerTopsis(pst, srcTable, dstTable, columns, mt);
        }
        catch(SQLException e)
        {
            e.printStackTrace();
            throw new RuntimeException(e);
        }
        finally
        {
            try
            {
                if(pst!=null) pst.close();
            }
            catch(SQLException e) {throw new RuntimeException(e);}
        }
        return mt;
    }
    //</editor-fold>
}
