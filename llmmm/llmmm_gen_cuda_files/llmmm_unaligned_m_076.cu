#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_76_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<76> mm_instantiator;

public:
  UnalignedM_76_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_76_MMInstantiatorWrapper__;

}  // namespace LLMMM
