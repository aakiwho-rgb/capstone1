import { cn } from "@/lib/utils";

export const BentoGrid = ({
  className,
  children,
}: {
  className?: string;
  children?: React.ReactNode;
}) => {
  return (
    <div
      className={cn(
        "mx-auto grid max-w-7xl grid-cols-1 gap-3 md:auto-rows-[14rem] md:grid-cols-3",
        className,
      )}
    >
      {children}
    </div>
  );
};

export const BentoGridItem = ({
  className,
  title,
  description,
  header,
  icon,
}: {
  className?: string;
  title?: string | React.ReactNode;
  description?: string | React.ReactNode;
  header?: React.ReactNode;
  icon?: React.ReactNode;
}) => {
  return (
    <div
      className={cn(
        "group/bento row-span-1 flex flex-col justify-between space-y-2 rounded-xl border border-border bg-card p-3 transition duration-200 hover:shadow-lg dark:hover:shadow-primary/5",
        className,
      )}
    >
      {header}
      <div className="transition duration-200 group-hover/bento:translate-x-1">
        <div className="flex items-center gap-2">
          {icon}
          <div className="font-medium text-sm text-card-foreground">
            {title}
          </div>
        </div>
        {description && (
          <div className="mt-1 text-xs text-muted-foreground line-clamp-2">
            {description}
          </div>
        )}
      </div>
    </div>
  );
};
