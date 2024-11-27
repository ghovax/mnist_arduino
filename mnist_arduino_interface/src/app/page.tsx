"use client"

import { useState, useEffect } from 'react'
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { AlertCircle, Camera, Link } from 'lucide-react'
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { useToast } from '@/hooks/use-toast'

export default function ArduinoCameraRecognition() {
  const [connectionStatus, setConnectionStatus] = useState<'disconnected' | 'connecting' | 'connected'>('disconnected')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [probabilities, setProbabilities] = useState<number[] | null>(null)
  const [predictedDigit, setPredictedDigit] = useState<number | null>(null)
  const [confidence, setConfidence] = useState<number | null>(null)

  const { toast } = useToast()

  const handleServerLogs = (logs: Array<{level: string, message: string, timestamp: string}>) => {
    logs.forEach(log => {
      const cleanMessage = log.message.replace(/\[\d+m/g, '').replace(/\u001b/g, '')
      const consoleMethod = log.level.toLowerCase() as keyof Console
      if (typeof console[consoleMethod] === 'function') {
        (console[consoleMethod] as (...args: any[]) => void)(`[${log.timestamp}] ${cleanMessage}`)
      } else {
        console.log(`[${log.timestamp}] ${log.level}: ${cleanMessage}`)
      }
    })
  }

  const connectArduino = async () => {
    setConnectionStatus('connecting')
    setError(null)

    try {
      const response = await fetch('/api/connect', { 
        method: 'POST',
        headers: {
          'Accept': 'application/json',
        }
      })
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      
      const data = await response.json()

      if (data.success) {
        setConnectionStatus('connected')
        toast({
          title: "Connected",
          description: `Successfully connected to port ${data.portName}`,
        })
      } else {
        throw new Error(data.errorMessage || 'Failed to connect')
      }

      if (data.logs) {
        handleServerLogs(data.logs)
      }
    } catch (error) {
      setConnectionStatus('disconnected')
      setError(`Connection error: ${error instanceof Error ? error.message : String(error)}`)
      toast({
        variant: "destructive",
        title: "Connection Failed",
        description: error instanceof Error ? error.message : String(error),
      })
    }
  }

  const captureImage = async () => {
    if (connectionStatus !== 'connected') {
      toast({
        variant: "destructive",
        title: "Not Connected",
        description: "Please connect to Arduino first",
      })
      return
    }

    setIsLoading(true)
    setError(null)
    setImageUrl(null)
    setPredictedDigit(null)
    setConfidence(null)
    setProbabilities(null)

    try {
      const response = await fetch('/api/capture', { 
        method: 'POST',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
        }
      })
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      
      const data = await response.json()
      
      if (data.logs) {
        handleServerLogs(data.logs)
      }

      console.log("Received response:", {
        success: data.success,
        hasImage: Boolean(data.originalImage),
        imageLength: data.originalImage?.length
      })

      if (data.success && data.originalImage) {
        const imageUrl = data.originalImage.startsWith('data:image') 
          ? data.originalImage 
          : `data:image/png;base64,${data.originalImage}`
        
        setImageUrl(imageUrl)
        setProbabilities(data.probabilities)
        setPredictedDigit(data.predictedDigit)
        setConfidence(data.confidence)

        toast({
          title: "Success",
          description: "Image captured and processed successfully",
        })
      } else {
        throw new Error(data.errorMessage || 'Failed to capture image')
      }
    } catch (error) {
      console.error("Capture error:", error)
      setError(`Capture error: ${error instanceof Error ? error.message : String(error)}`)
      toast({
        variant: "destructive",
        title: "Capture Failed",
        description: error instanceof Error ? error.message : String(error),
      })
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="container min-w-[640px] max-w-2xl mx-auto px-4 py-6">
      {/* Header Section */}
      <div className="mb-6">
        <h1 className="text-xl font-semibold mb-1">
          Arduino MNIST Recognition
        </h1>
        <p className="text-sm text-muted-foreground">
          Capture and recognize handwritten digits using Arduino camera and TensorFlow
        </p>
      </div>

      {/* Control Panel */}
      <Card className="mb-6 min-w-[640px]">
        <CardHeader className="py-3">
          <CardTitle className="flex items-center gap-2 text-sm">
            <div className="h-2 w-2 rounded-full" 
                 style={{ backgroundColor: connectionStatus === 'connected' ? '#16a34a' : '#dc2626' }} />
            Device Status
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 py-0 pb-3">
          <div className="flex gap-2">
            <Button 
              onClick={connectArduino} 
              disabled={connectionStatus === 'connecting'}
              variant={connectionStatus === 'connected' ? "secondary" : "default"}
              className="flex-1"
              size="sm"
            >
              <Link className="mr-2 h-3 w-3" />
              {connectionStatus === 'connected' ? 'Reconnect Device' : 'Connect Device'}
            </Button>
            <Button 
              onClick={captureImage} 
              disabled={connectionStatus !== 'connected' || isLoading}
              className="flex-1"
              size="sm"
            >
              <Camera className="mr-2 h-3 w-3" />
              {isLoading ? 'Capturing...' : 'Capture Image'}
            </Button>
          </div>

          {/* Status Messages */}
          {connectionStatus !== 'connected' && (
            <Alert variant="destructive" className="py-2">
              <AlertCircle className="h-3 w-3" />
              <AlertTitle className="text-sm">Not Connected</AlertTitle>
              <AlertDescription className="text-xs">
                Please connect to Arduino to start capturing images
              </AlertDescription>
            </Alert>
          )}

          {error && (
            <Alert variant="destructive" className="py-2">
              <AlertCircle className="h-3 w-3" />
              <AlertTitle className="text-sm">Error</AlertTitle>
              <AlertDescription className="text-xs">{error}</AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Results Section */}
      <Card className="min-w-[640px]">
        <CardHeader className="py-3">
          <CardTitle className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-2">
              <Camera className="h-3 w-3" />
              Results
            </div>
            {predictedDigit !== null && confidence !== null && (
              <div className="flex items-center gap-2 text-xs">
                <span className="font-medium text-base">Predicted digit: {predictedDigit}</span>
                <div className="h-1 w-24 rounded-full bg-secondary">
                  <div 
                    className="h-full rounded-full bg-foreground"
                    style={{ width: `${confidence * 100}%` }}
                  />
                </div>
                <span className="text-muted-foreground">
                  ({(confidence * 100).toFixed(1)}%)
                </span>
              </div>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent className="p-3">
          <div className="flex gap-6">
            {/* Image Section */}
            <div className="w-40">
              <div className="mb-2">
                <span className="text-xs font-medium text-muted-foreground">Captured Image</span>
              </div>
              <div className="border bg-secondary/50 flex items-center justify-center overflow-hidden">
                {imageUrl ? (
                  <img 
                    src={imageUrl} 
                    alt="Captured digit" 
                    className="object-contain w-full image-pixelated"
                    onError={(e) => {
                      console.error("Image failed to load:", e);
                      setError("Failed to load image");
                    }}
                  />
                ) : (
                  <p className="text-muted-foreground text-sm text-center p-4">
                    {isLoading ? (
                      <span className="flex items-center gap-2">
                        <span className="h-3 w-3 border-2 border-muted-foreground border-r-transparent rounded-full animate-spin" />
                        Processing...
                      </span>
                    ) : (
                      "No image captured yet"
                    )}
                  </p>
                )}
              </div>
            </div>

            {/* Distribution Section */}
            <div className="flex-1">
              {predictedDigit !== null && probabilities ? (
                <div className="space-y-2">
                  <span className="text-xs font-medium text-muted-foreground">Probability Distribution</span>
                  <div className="h-52 grid grid-cols-10 gap-px">
                    {probabilities.map((prob, index) => (
                      <div 
                        key={index} 
                        className="relative flex flex-col h-full"
                      >
                        <div className="flex-1 relative bg-secondary">
                          <div 
                            className={`absolute bottom-0 left-0 right-0 ${prob > 0.05 ? 'bg-foreground' : 'bg-transparent'}`}
                            style={{ height: `${prob * 100}%` }} 
                          />
                        </div>
                        <span className="text-center text-xs text-muted-foreground mt-1">
                          {index}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="h-full flex items-center justify-center border bg-secondary/50">
                  <p className="text-sm text-muted-foreground">
                    No prediction
                  </p>
                </div>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

