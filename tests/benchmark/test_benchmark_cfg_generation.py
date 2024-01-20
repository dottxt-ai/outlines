import pytest

import outlines.grammars

# source: https://json.org/example.html
json_sample = """{"web-app": {
  "servlet": [
    {
      "servlet-name": "cofaxCDS",
      "servlet-class": "org.cofax.cds.CDSServlet",
      "init-param": {
        "configGlossary:installationAt": "Philadelphia, PA",
        "configGlossary:adminEmail": "ksm@pobox.com",
        "configGlossary:poweredBy": "Cofax",
        "configGlossary:poweredByIcon": "/images/cofax.gif",
        "configGlossary:staticPath": "/content/static",
        "templateProcessorClass": "org.cofax.WysiwygTemplate",
        "templateLoaderClass": "org.cofax.FilesTemplateLoader",
        "templatePath": "templates",
        "templateOverridePath": "",
        "defaultListTemplate": "listTemplate.htm",
        "defaultFileTemplate": "articleTemplate.htm",
        "useJSP": false,
        "jspListTemplate": "listTemplate.jsp",
        "jspFileTemplate": "articleTemplate.jsp",
        "cachePackageTagsTrack": 200,
        "cachePackageTagsStore": 200,
        "cachePackageTagsRefresh": 60,
        "cacheTemplatesTrack": 100,
        "cacheTemplatesStore": 50,
        "cacheTemplatesRefresh": 15,
        "cachePagesTrack": 200,
        "cachePagesStore": 100,
        "cachePagesRefresh": 10,
        "cachePagesDirtyRead": 10,
        "searchEngineListTemplate": "forSearchEnginesList.htm",
        "searchEngineFileTemplate": "forSearchEngines.htm",
        "searchEngineRobotsDb": "WEB-INF/robots.db",
        "useDataStore": true,
        "dataStoreClass": "org.cofax.SqlDataStore",
        "redirectionClass": "org.cofax.SqlRedirection",
        "dataStoreName": "cofax",
        "dataStoreDriver": "com.microsoft.jdbc.sqlserver.SQLServerDriver",
        "dataStoreUrl": "jdbc:microsoft:sqlserver://LOCALHOST:1433;DatabaseName=goon",
        "dataStoreUser": "sa",
        "dataStorePassword": "dataStoreTestQuery",
        "dataStoreTestQuery": "SET NOCOUNT ON;select test='test';",
        "dataStoreLogFile": "/usr/local/tomcat/logs/datastore.log",
        "dataStoreInitConns": 10,
        "dataStoreMaxConns": 100,
        "dataStoreConnUsageLimit": 100,
        "dataStoreLogLevel": "debug",
        "maxUrlLength": 500}},
    {
      "servlet-name": "cofaxEmail",
      "servlet-class": "org.cofax.cds.EmailServlet",
      "init-param": {
      "mailHost": "mail1",
      "mailHostOverride": "mail2"}},
    {
      "servlet-name": "cofaxAdmin",
      "servlet-class": "org.cofax.cds.AdminServlet"},

    {
      "servlet-name": "fileServlet",
      "servlet-class": "org.cofax.cds.FileServlet"},
    {
      "servlet-name": "cofaxTools",
      "servlet-class": "org.cofax.cms.CofaxToolsServlet",
      "init-param": {
        "templatePath": "toolstemplates/",
        "log": 1,
        "logLocation": "/usr/local/tomcat/logs/CofaxTools.log",
        "logMaxSize": "",
        "dataLog": 1,
        "dataLogLocation": "/usr/local/tomcat/logs/dataLog.log",
        "dataLogMaxSize": "",
        "removePageCache": "/content/admin/remove?cache=pages&id=",
        "removeTemplateCache": "/content/admin/remove?cache=templates&id=",
        "fileTransferFolder": "/usr/local/tomcat/webapps/content/fileTransferFolder",
        "lookInContext": 1,
        "adminGroupID": 4,
        "betaServer": true}}],
  "servlet-mapping": {
    "cofaxCDS": "/",
    "cofaxEmail": "/cofaxutil/aemail/*",
    "cofaxAdmin": "/admin/*",
    "fileServlet": "/static/*",
    "cofaxTools": "/tools/*"},

  "taglib": {
    "taglib-uri": "cofax.tld",
    "taglib-location": "/WEB-INF/tlds/cofax.tld"}}}"""


csv_sample = """ID,Name,Age,FavFruit
1,Andrew,30,Banana
2,Mohammad,40,Apple
3,Alice,61,Peach"""


yaml_sample = """version: 2

python:
  version: "3.8"
  install:
      - method: pip
        path: .
        extra_requirements:
          - rtd
      - requirements: requirements-doc.txt

sphinx:
  builder: html
  configuration: docs/source/conf.py
  fail_on_warning: true
"""


python3_sample = """
import os
from functools import lru_cache

foo = lambda x: print(x)

@pytest.mark.parametrize("schema_name", schemas.keys())
def test_benchmark_json_schema_to_regex(benchmark, ensure_numba_compiled, schema_name):
    schema = schemas[schema_name]
    benchmark.pedantic(
        build_regex_from_object,
        args=(schema,),
        rounds=8,
    )
"""


lisp_sample = (
    """(defun plswork () (format t "Geeze I sure hope this grammar works!"))"""
)


"""
SQLite Samples derived from https://github.com/kacos2000/Queries
https://github.com/kacos2000/Queries License:
MIT License

Copyright (c) 2018, 2019 Costas Katsavounidis
https://linkedin.com/in/kacos2000
Source: https://github.com/kacos2000/queries/


Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
sqlite_sample_AppStateChangeSummary = r"""
-- Diagnostic
--           KernelProcess.AppStateChangeSummary
-- from C:\ProgramData\Microsoft\Diagnosis\EventTranscript\EventTranscript.db

SELECT

--Timestamp from db field
json_extract(events_persisted.payload,'$.time') as 'UTC TimeStamp',

-- Timestamp from json payload
datetime((timestamp - 116444736000000000)/10000000, 'unixepoch','localtime') as 'Local TimeStamp',
json_extract(events_persisted.payload,'$.ext.loc.tz') as 'TimeZome',
json_extract(events_persisted.payload,'$.ext.utc.seq') as 'seq',

-- Event
replace( events_persisted.full_event_name,'KernelProcess.AppStateChangeSummary','') as 'Event',

-- Counters
json_extract(events_persisted.payload,'$.data.LaunchCount') as 'LaunchCount',
json_extract(events_persisted.payload,'$.data.SuspendCount') as 'SuspendCount',
json_extract(events_persisted.payload,'$.data.ResumeCount') as 'ResumeCount',
json_extract(events_persisted.payload,'$.data.TerminateCount') as 'TerminateCount',
json_extract(events_persisted.payload,'$.data.CrashCount') as 'CrashCount',


-- Target App Info
case json_extract(events_persisted.payload,'$.data.TargetAppType')
    when 'Modern' then json_extract(events_persisted.payload,'$.data.TargetAppType')||" (UWP)"
    when 'Desktop' then json_extract(events_persisted.payload,'$.data.TargetAppType')||" (Win)"
     else json_extract(events_persisted.payload,'$.data.TargetAppType')
    end as 'TargetAppIdType',

-- Target Application Name
case when substr(json_extract(events_persisted.payload,'$.data.TargetAppId'),1,1) is 'W' -- Windows Application x32/x64
    then substr(json_extract(events_persisted.payload,'$.data.TargetAppId'),93)
    else substr(json_extract(events_persisted.payload,'$.data.TargetAppId'),3)
    end as 'TargetAppId Name',

-- SHA1 Hash of the application that produced this event
case when substr(json_extract(events_persisted.payload,'$.data.TargetAppId'),1,1) is 'W' -- Windows Application x32/x64
    then upper(substr(json_extract(events_persisted.payload,'$.data.TargetAppId'),52,40 ))
    -- Same as the 'FileId' in Amcache.hve (Root\InventoryApplicationFile\)
    end as 'Target AppId SHA1',    -- (SHA1 Base16) checked & verified

-- ProgramId of the application that produced this event
case when substr(json_extract(events_persisted.payload,'$.data.TargetAppId'),1,1) is 'W' -- Windows Application x32/x64
    then upper(substr(json_extract(events_persisted.payload,'$.data.TargetAppId'),3,44 ))
    end as 'Target AppId ProgramId',   -- Same as the 'ProgramId' in Amcache.hve (Root\InventoryApplicationFile\)

-- Universal Windows Platform version info
case when substr(json_extract(events_persisted.payload,'$.data.TargetAppId'),1,1) is 'W'
    then substr(json_extract(events_persisted.payload,'$.data.TargetAppVer'),(instr(substr(json_extract(events_persisted.payload,'$.data.TargetAppVer'),22),'!')+22))
    else substr(json_extract(events_persisted.payload,'$.data.TargetAppVer'),(instr(substr(json_extract(events_persisted.payload,'$.data.TargetAppVer'),22),'!%!')))
    end as 'TargetApp Ver',


-- Local, MS or AAD account
trim(json_extract(events_persisted.payload,'$.ext.user.localId'),'m:') as 'UserId',
sid as 'User SID',

logging_binary_name

from events_persisted
where events_persisted.full_event_name like 'KernelProcess.AppStateChangeSummary%'

 -- Sort by event date dscending (newest first)
order by cast(events_persisted.timestamp as integer) desc
"""


sqlite_sample_Census = r"""
SELECT

--Timestamp from db field
json_extract(events_persisted.payload,'$.time') as 'UTC TimeStamp',
-- Timestamp from json payload
datetime((timestamp - 116444736000000000)/10000000, 'unixepoch','localtime') as 'Local TimeStamp',
json_extract(events_persisted.payload,'$.ext.loc.tz') as 'TimeZome',
json_extract(events_persisted.payload,'$.ext.utc.seq') as 'seq',

-- events
replace(events_persisted.full_event_name,'Census.','') as 'Event',
--replace(json_extract(events_persisted.payload,'$.name'),'Census.','') as 'Name',

-- Current state / Settings
json_extract(events_persisted.payload,'$.data') as 'State \ Settings',


-- Local, MS or AAD account
trim(json_extract(events_persisted.payload,'$.ext.user.localId'),'m:') as 'UserId',
sid as 'User SID',

logging_binary_name


from events_persisted
where
-- include events:
  events_persisted.full_event_name like 'Census%'



 -- Sort by event datedescending (newest first)
order by cast(events_persisted.timestamp as integer) desc
"""


all_samples = {
    "json": (outlines.grammars.json, json_sample),
    "csv": (outlines.grammars.csv, csv_sample),
    "yaml": (outlines.grammars.yaml, yaml_sample),
    "python3": (outlines.grammars.python3, python3_sample),
    "lisp": (outlines.grammars.lisp, lisp_sample),
    "sqlite_AppStateChangeSummary": (
        outlines.grammars.sqlite,
        sqlite_sample_AppStateChangeSummary,
    ),
    "sqlite_Census": (outlines.grammars.sqlite, sqlite_sample_Census),
}


class MockEngine:
    def __init__(self, logits_processor, to_generate):
        self.logits_processor = logits_processor
        self.to_generate = to_generate

        self._generated_token_ids = []

        # precompute legal tokens at each step to ensure we're
        # only measuring the performance of the logits processor
        import pdb

        pdb.set_trace()

    def run_until_eos(self):
        while self.to_generate:
            logits = self.logits_processor(self._mock_logits, self._generated_token_ids)


@pytest.mark.parametrize("sample_name", all_samples.keys())
def test_benchmark_cfg_generation(
    benchmark, tokenizer, ensure_numba_compiled, sample_name
):
    """Benchmark CFGLogitsProcessor Generation"""

    cfg, sample = all_samples[sample_name]
    cfg_logits_processor = CFGLogitsProcessor(cfg, tokenizer)

    engine = MockEngine(
        logits_processor=cfg_logits_processor,
        to_generate=sample,
    )

    benchmark(
        engine.run_until_eos,
    )
