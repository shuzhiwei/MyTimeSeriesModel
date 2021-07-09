package com.hlzd.timeseries

import java.sql.Timestamp
import java.time.{ZoneId, ZonedDateTime}

import com.cloudera.sparkts._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row, SQLContext, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}

/**
 * 时间序列模型time-series的建立
 * @author shuzhiwei
 * @create 2021-07-09 15:43
 */
object TimeSeriesTrain {
  /**
   * 把数据中的“time”列转换成固定时间格式：ZonedDateTime（such as 2007-12-03T10:15:30+01:00 Europe/Paris.）
   * @param timeDataKeyDf
   * @param sqlContext
   * @param hiveColumnName
   * @return zonedDateDataDf
   */
  def timeChangeToDate(timeDataKeyDf:DataFrame,
                       sqlContext: SQLContext,
                       hiveColumnName:List[String],
                       startTime:String,
                       sc:SparkContext): DataFrame ={
    var rowRDD:RDD[Row]=sc.parallelize(Seq(Row(""),Row("")))
    //具体到月份
    if(startTime.length==6){
      rowRDD=timeDataKeyDf.rdd.map{row=>
        row match{
          case Row(time,key,data)=>{
            val dt = ZonedDateTime.of(time.toString.substring(0,4).toInt,time.toString.substring(4).toInt,1,0,0,0,0,ZoneId.systemDefault())
            Row(Timestamp.from(dt.toInstant), key.toString, data.toString.toDouble)
          }
        }
      }
    }else if(startTime.length==8){
      //具体到日
      rowRDD=timeDataKeyDf.rdd.map{row=>
        row match{
          case Row(time,key,data)=>{
            val dt = ZonedDateTime.of(time.toString.substring(0,4).toInt,time.toString.substring(4,6).toInt,time.toString.substring(6).toInt,0,0,0,0,ZoneId.systemDefault())
            Row(Timestamp.from(dt.toInstant), key.toString, data.toString.toDouble)
          }
        }
      }
    }
    //根据模式字符串生成模式，转化成dataframe格式
    val field=Seq(
      StructField(hiveColumnName(0), TimestampType, true),
      StructField(hiveColumnName(0)+"Key", StringType, true),
      StructField(hiveColumnName(1), DoubleType, true)
    )
    val schema=StructType(field)
    val zonedDateDataDf=sqlContext.createDataFrame(rowRDD,schema)
    return zonedDateDataDf
  }


  /**
   * 总方法调用
   * @param args
   */
  def main(args: Array[String]) {
    /** ***环境设置 *****/
    //shield the unnecessary log in terminal
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    //set the environment
//    System.setProperty("hadoop.home.dir", "D:\\ideaIU\\hadoop-2.2.0-x64-bin\\")
    val sparkConf = new SparkConf().setAppName("TimeSeriesTrain").setMaster("local[*]")
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()
    val sc = spark.sparkContext

    /** ***参数设置 *****/
    //选择模型(holtwinters或者是arima)
    val modelName = "holtwinters"
    //日期的开始和结束，格式为“yyyyMM”或者为“yyyyMMdd”
    val startTime = "200305"
    val endTime = "201412"
    //预测后面N个值
    val predictedN = 19

//    /*****参数设置*****/
//    //hive中的数据库名字.数据表名
//    val databaseTableName="time_series.jxt_electric_month"
//    //选择模型(holtwinters或者是arima)
//    val modelName="holtwinters"
//    //选择要hive的数据表中要处理的time和data列名（输入表中用于训练的列名,必须前面是时间，后面是data）
//    val hiveColumnName=List("time","data")
//    //日期的开始和结束，格式为“yyyyMM”或者为“yyyyMMdd”
//    val startTime="200305"
//    val endTime="201412"
//    //预测后面N个值
//    val predictedN=19
//    //存放的表名字
//    val outputTableName="timeseries_outputdate"

    //只有holtWinters才有的参数
    //季节性参数（12或者4）
    val period = 12
    //holtWinters选择模型：additive（加法模型）、Multiplicative（乘法模型）
    val holtWintersModelType = "Multiplicative"

    /** ***读取数据和创建训练数据 *****/
    //    //read the data form the hive
    //    val hiveDataDf=hiveContext.sql("select * from "+databaseTableName)
    //      .select(hiveColumnName.head,hiveColumnName.tail:_*)
    val hiveDataDf = spark.read.format("csv")
      .option("header", "true")
      .option("multiLine", "true")
      .load("F:\\test\\MyTimeSeriesModel\\src\\main\\resources\\ceshi.csv")


    //In hiveDataDF:increase a new column.This column's name is hiveColumnName(0)+"Key",it's value is 0.
    //The reason is:The string column labeling which string key the observation belongs to.
    val zonedDateDataDf = timeChangeToDate(hiveDataDf, spark, startTime, sc)

    /**
     * 创建数据中时间的跨度（Create an daily DateTimeIndex）:开始日期+结束日期+递增数
     * 日期的格式要与数据库中time数据的格式一样
     */
    //参数初始化
    val zone = ZoneId.systemDefault()
    var dtIndex: UniformDateTimeIndex = DateTimeIndex.uniformFromInterval(
      ZonedDateTime.of(2003, 1, 1, 0, 0, 0, 0, zone),
      ZonedDateTime.of(2004, 1, 1, 0, 0, 0, 0, zone),
      new MonthFrequency(1))

    //具体到月份
    if (startTime.length == 6) {
      dtIndex = DateTimeIndex.uniformFromInterval(
        ZonedDateTime.of(startTime.substring(0, 4).toInt, startTime.substring(4).toInt, 1, 0, 0, 0, 0, zone),
        ZonedDateTime.of(endTime.substring(0, 4).toInt, endTime.substring(4).toInt, 1, 0, 0, 0, 0, zone),
        new MonthFrequency(1))
    } else if (startTime.length == 8) {
      //具体到日,则把dtIndex覆盖了
      dtIndex = DateTimeIndex.uniformFromInterval(
        ZonedDateTime.of(startTime.substring(0, 4).toInt, startTime.substring(4, 6).toInt, startTime.substring(6).toInt, 0, 0, 0, 0, zone),
        ZonedDateTime.of(endTime.substring(0, 4).toInt, endTime.substring(4, 6).toInt, endTime.substring(6).toInt, 0, 0, 0, 0, zone),
        new DayFrequency(1))
    }

    //创建训练数据TimeSeriesRDD(key,DenseVector(series))
    val trainTsrdd = TimeSeriesRDD.timeSeriesRDDFromObservations(dtIndex, zonedDateDataDf,
      hiveColumnName(0), hiveColumnName(0) + "Key", hiveColumnName(1))

    /** ***建立Modle对象 *****/
    val timeSeriesModel = new TimeSeriesModel(predictedN, outputTableName)
    var forecastValue: RDD[String] = sc.parallelize(Seq(""))
    //选择模型
    modelName match {
      case "arima" => {
        //创建和训练arima模型
        forecastValue = timeSeriesModel.arimaModelTrain(trainTsrdd)
      }
      case "holtwinters" => {
        //创建和训练HoltWinters模型(季节性模型)
        forecastValue = timeSeriesModel.holtWintersModelTrain(trainTsrdd, period, holtWintersModelType)
      }
      case _ => throw new UnsupportedOperationException("Currently only supports 'ariam' and 'holtwinters")
    }

    //合并实际值和预测值，并加上日期,形成dataframe(Date,Data)，并保存
    timeSeriesModel.actualForcastDateSaveInHive(trainTsrdd, forecastValue, modelName, predictedN, startTime, endTime, sc, hiveColumnName, sqlContext)
  }
}
