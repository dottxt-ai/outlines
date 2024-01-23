-- SQLite Samples derived from https://github.com/kacos2000/Queries
-- https://github.com/kacos2000/Queries License:
-- MIT License
--
-- Copyright (c) 2018, 2019 Costas Katsavounidis
-- https://linkedin.com/in/kacos2000
-- Source: https://github.com/kacos2000/queries/
--
--
-- Permission is hereby granted, free of charge, to any person obtaining a copy
-- of this software and associated documentation files (the "Software"), to deal
-- in the Software without restriction, including without limitation the rights
-- to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
-- copies of the Software, and to permit persons to whom the Software is
-- furnished to do so, subject to the following conditions:
--
-- The above copyright notice and this permission notice shall be included in all
-- copies or substantial portions of the Software.
--
-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
-- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
-- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
-- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
-- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
-- OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
-- SOFTWARE.

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
