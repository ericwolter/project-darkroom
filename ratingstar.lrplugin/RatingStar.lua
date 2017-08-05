require "Constants"
require "Utils"

local LrLogger = import 'LrLogger'
local LrApplication = import 'LrApplication'
local LrTasks = import 'LrTasks'
local LrFileUtils = import 'LrFileUtils'
local LrPathUtils = import 'LrPathUtils'
local LrHttp = import 'LrHttp'
local LrProgressScope = import 'LrProgressScope'

local myLogger = LrLogger( 'exportLogger' )
myLogger:enable( "logfile" )

local thumbnailDir = LrPathUtils.getStandardFilePath('temp')

RatingStar = {
  killTask = false,
  shutdown = false
}

-- catalog:withWriteAccessDo('Set rating', function(context)
--   photo:setRawMetadata('rating', rating)
-- end )

function RatingStar:requestScores(progress, catalog, target_photos, callback)
  -- LrTasks.sleep(5)
  -- LrTasks.yield()
  if self.killTask or self.shutdown then
    callback(nil)
    return
  end

  local count = #target_photos
  progress:setPortionComplete(54-count, 54)

  myLogger:trace('Count:', count)
  if count == 0 then
    callback(nil)
    return
  end

-- https://forums.adobe.com/thread/1594284
  local photo = table.remove(target_photos)

  local f = function(data, err)
    if err == nil then
      local filepath = photo.path
      local filename = LrPathUtils.leafName(filepath)
      local path = LrPathUtils.child(thumbnailDir, filename)
      local jpgpath = LrPathUtils.addExtension(path, 'jpg')
      myLogger:trace('Got jpeg thumbnail typeof', type(data))
      myLogger:trace(data)
      myLogger:trace('Saving thumbnail to', jpgpath)
      local f = io.open(jpgpath, 'wb')
      f:write(data)
      f:close()

      local url = CLIENT_ENDPOINT..urlencode(jpgpath)
      myLogger:trace('Request score', url)
      res, info = LrHttp.get(url, nil, 10)
      score = tonumber(res)
      rating = score2rating(score)
      catalog:withWriteAccessDo('Set rating', function(context)
        photo:setRawMetadata('rating', rating)
      end )
      LrFileUtils.delete(jpgpath)
      self:requestScores(progress, catalog, target_photos, callback)
      -- myLogger:trace('Score:', score, rating)
    else
      myLogger:error(err)
    end
  end

  photo:requestJpegThumbnail(299, 299, f)
end

function RatingStar:init()
  LrTasks.startAsyncTask( function()

    -- LrTasks.startAsyncTask( function()
    --   LrTasks.execute('python C:\\Users\\Home\\Documents\\GitHub\\project-darkroom\\ratingstar.py')
    -- end )

    local catalog = LrApplication.activeCatalog()
    local photos = catalog:getTargetPhotos()
    local progress = LrProgressScope({caption="Rating photos"})
    self:requestScores(progress, catalog, photos, function(err)
      myLogger:trace('Done!')
    end)
  end, "Client Start" )
end
