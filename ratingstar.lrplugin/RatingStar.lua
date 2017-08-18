local Require = require "Require".path ("../debuggingtoolkit.lrdevplugin").reload()
local Debug = require "Debug"
require "strict"

require "Constants"
require "Utils"

local logging = require "Logging.lua"
local LrApplication = import 'LrApplication'
local LrTasks = import 'LrTasks'
local LrFileUtils = import 'LrFileUtils'
local LrPathUtils = import 'LrPathUtils'
local LrHttp = import 'LrHttp'
local LrProgressScope = import 'LrProgressScope'

local thumbnailDir = LrPathUtils.getStandardFilePath('temp')

RatingStar = {
  killTask = false,
  shutdown = false
}

function RatingStar:requestScores(progress, count, target_photos, callback)
  LrTasks.yield()
  LrTasks.sleep(1)
  if self.killTask or self.shutdown then
    callback(nil)
    return
  end

  if progress.parent_scope:isCanceled() then
    callback(nil)
    return
  end

  -- #local count = #target_photos
  progress.parent_scope:setPortionComplete(count, progress.total_count)

  logging:log("working on photo "..tostring(count)..".")
  if count == 0 then
    callback(nil)
    return
  end

-- https://forums.adobe.com/thread/1594284
  local photo = target_photos[count]
  logging:log("got photo "..tostring(photo.localIdentifier)..".")

  local f = function(data, err)
    if data ~= nil then
      -- local filepath = photo.path
      -- logging:log("photo is located at "..filepath)
      -- local filename = LrPathUtils.leafName(filepath)
      -- logging:log("photo file name is "..filename)
      -- local path = LrPathUtils.child(thumbnailDir, filename)
      -- local jpgpath = LrPathUtils.addExtension(path, 'jpg')
      -- logging:log("saving thumbnail to "..jpgpath)
      -- local f = io.open(jpgpath, 'wb')
      -- f:write(data)
      -- f:close()
      --
      -- local url = CLIENT_ENDPOINT..urlencode(jpgpath)
      -- logging:log("requesting score at "..url)
      -- Debug.pauseIfAsked()
      -- local res, info = LrHttp.get(url)
      -- logging:log("received response with status "..tostring(info.status))
      -- local score = tonumber(res)
      -- local rating = score2rating(score)
      -- logging:log("trying to write rating "..tostring(rating).." to catalog.")
      -- photo.catalog:withWriteAccessDo('Set rating', Debug.showErrors(function(context)
      --   photo:setRawMetadata('rating', rating)
      -- end ))
      -- logging:log("wrote rating to catalog was successful.")
      --
      -- LrFileUtils.delete(jpgpath)
      -- logging:log("deleted temporary thumbnail.")
      self:requestScores(progress, count+1, target_photos, callback)
    else
      logging:log("error: failed to request score for photo "..tostring(photo.localIdentifier).." with message '"..tostring(err).."")
    end
  end

  photo:requestJpegThumbnail(299, 299, f)
end

function RatingStar:init()
  LrTasks.startAsyncTask(Debug.showErrors(function()
    local catalog = LrApplication.activeCatalog()
    logging:log("got active catalog.")
    local photos = catalog:getTargetPhotos()
    local totalCount = #photos
    logging:log("got "..tostring(totalCount).." target photos.")
    local progress = {
      parent_scope = LrProgressScope( { titel="Rating photos" } ),
      total_count = totalCount
    }

    -- for count, photo in ipairs(photos) do
    --     if self.killTask or self.shutdown then
    --       break
    --     end
    --
    --     if progress.parent_scope:isCanceled() then
    --       break
    --     end
    --
    --     progress.parent_scope:setPortionComplete(count, progress.total_count)
    --     logging:log("working on photo "..tostring(count)..".")
    --
    --     logging:log("got photo "..tostring(photo.localIdentifier)..".")
    --     photo:requestJpegThumbnail(299, 299, function(data, err)
    --       if err ~= nil then
    --         logging:log("error: failed to request score for photo "..tostring(photo.localIdentifier).." with message '"..tostring(err).."")
    --       else
    --         local filepath = photo.path
    --         logging:log("photo is located at "..filepath)
    --         local filename = LrPathUtils.leafName(filepath)
    --         logging:log("photo file name is "..filename)
    --         local path = LrPathUtils.child(thumbnailDir, filename)
    --         local jpgpath = LrPathUtils.addExtension(path, 'jpg')
    --         logging:log("saving thumbnail to "..jpgpath)
    --         -- local f = io.open(jpgpath, 'wb')
    --         -- f:write(data)
    --         -- f:close()
    --         --
    --         -- local url = CLIENT_ENDPOINT..urlencode(jpgpath)
    --         -- logging:log("requesting score at "..url)
    --         -- local res, info = LrHttp.get(url, nil, 1)
    --         -- logging:log("received response with status "..tostring(info.status))
    --         -- local score = tonumber(res)
    --         -- local rating = score2rating(score)
    --         -- logging:log("trying to write rating "..tostring(rating).." to catalog.")
    --         -- photo.catalog:withWriteAccessDo('Set rating', Debug.showErrors(function(context)
    --         --   photo:setRawMetadata('rating', rating)
    --         -- end ))
    --         -- logging:log("wrote rating to catalog was successful.")
    --         --
    --         -- LrFileUtils.delete(jpgpath)
    --         -- logging:log("deleted temporary thumbnail.")
    --       end
    --     end )
    -- end
    --
    -- logging:log("completed rating photos.")

    self:requestScores(progress, 1, photos, function(err)
      progress.parent_scope:done()
      logging:log("completed rating photos.")
    end)
  end, "Client Start" ))
end
